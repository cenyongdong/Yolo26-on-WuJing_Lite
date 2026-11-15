# 在已链接设计检查约束是否实际生效；布局布线后还检查每个端点的真实 slack。
# 预布局可设置 y26_check_slack=0；绝不能把预布局验证写作物理时序通过。
if {![info exists y26_check_slack]} {set y26_check_slack 1}
if {$y26_check_slack} {set y26_cdc_report_only 1}
source [file join [file dirname [info script]] y26_soc_cdc.tcl]
unset -nocomplain y26_cdc_report_only
set y26_report [open [file join $out yolo26-cdc-endpoints.tsv] w]
puts $y26_report "group\tendpoint\trequirement_ns\tslack_ns"
set y26_failed 0
foreach group {request response command result stop link} period {5 10 5 10 5 10} {
    set sources [y26_cdc_pins [set y26_${group}_source] C]
    set targets [y26_cdc_pins [set y26_${group}_target] D]
    set paths [get_timing_paths -delay_type max -from $sources -to $targets -nworst 1 -max_paths [llength $targets]]
    set visited [dict create]
    foreach path $paths {
        set endpoint [get_property ENDPOINT_PIN $path]
        set requirement [get_property REQUIREMENT $path]
        set slack [get_property SLACK $path]
        dict set visited $endpoint 1
        puts $y26_report "$group\t$endpoint\t$requirement\t$slack"
        if {abs($requirement-$period)>0.001 || ($y26_check_slack && $slack<0)} {set y26_failed 1}
    }
    foreach endpoint $targets {
        if {![dict exists $visited [get_property NAME $endpoint]]} {
            puts $y26_report "$group\t$endpoint\tMISSING\tMISSING"
            set y26_failed 1
        }
    }
}
close $y26_report
if {$y26_failed} {error "YOLO26 CDC constraint coverage or slack failed; endpoint report preserved"}

if {$y26_check_slack} {
    # 物理优化可能把命令高度字段的两级接收寄存器装入 DSP48 的 B 端口。
    set bytes_dsp [get_cells -quiet U0_ps_ai_peripheral/yolo26_video/preprocess/engine/expected_source_bytes_reg]
    set dsp_report [open $out/yolo26-dsp-cdc.tsv w]
    if {[llength $bytes_dsp] == 1 && [get_property REF_NAME $bytes_dsp] eq "DSP48E1"} {
        puts $dsp_report "BREG\t[get_property BREG $bytes_dsp]"
        puts $dsp_report "CLK\t[get_clocks -of_objects [get_pins $bytes_dsp/CLK]]"
        foreach enable {CEB1 CEB2} {
            set drivers [all_fanin -flat -startpoints_only -to [get_pins $bytes_dsp/$enable]]
            puts $dsp_report "$enable\t$drivers"
        }
        for {set bit 0} {$bit < 16} {incr bit} {
            set source [get_pins [format {U0_ps_ai_peripheral/yolo26_video/preprocess/control/mailbox/payload_hold_reg[%d]/C} [expr {144+$bit}]]]
            set sink [get_pins [format {%s/B[%d]} $bytes_dsp $bit]]
            if {$source ni [all_fanin -flat -startpoints_only -to $sink]} {continue}
            set path [get_timing_paths -quiet -delay_type max -from $source -to $sink -max_paths 1]
            if {[llength $path] != 1 || abs([get_property REQUIREMENT $path] - 5) > 0.001 ||
                [get_property SLACK $path] < 0} {
                close $dsp_report
                error "Unproved held-payload DSP path: $source to $sink"
            }
            puts $dsp_report "PATH\t$source\t$sink\t[get_property REQUIREMENT $path]\t[get_property SLACK $path]"
        }
    }
    close $dsp_report
}
