# GP=100MHz，NPU 外部访存接口=既有 MIG clk_pll_i 200MHz。绑定真实寄存器时钟，不新建/修改时钟。
# 两方向保持载荷先于三拍通知到达；不使用 false path，不修改既有 AI 最小周期例外。
proc trace_cells {suffix} {
    set result [get_cells -hier -regexp "^U0_ps_ai_peripheral/npu_trace/control_mailbox/$suffix"]
    if {![llength $result]} {error "NPU trace CDC registers missing: $suffix"}
    return $result
}
proc trace_pins {cells ref} {
    set pins [get_pins -of_objects $cells -filter "REF_PIN_NAME == $ref"]
    if {[llength $pins] != [llength $cells]} {error "NPU trace CDC pin coverage differs: $ref"}
    return $pins
}
set trace_request_source [trace_cells {request_toggle_reg$}]
set trace_request_target [trace_cells {request_sync_reg\[0\]$}]
set trace_response_source [trace_cells {response_toggle_reg$}]
set trace_response_target [trace_cells {response_sync_reg\[0\]$}]
set trace_command_source [trace_cells {payload_hold_reg\[[0-9]+\]$}]
set trace_command_target [trace_cells {target_command_reg\[[0-9]+\]$}]
set trace_result_source [trace_cells {response_hold_reg\[[0-9]+\]$}]
set trace_result_target [trace_cells {source_response_reg\[[0-9]+\]$}]
if {![info exists trace_check_slack]} {set trace_check_slack 1}
set trace_report [open [file join $out npu-trace-cdc-endpoints.tsv] w]
puts $trace_report "group\tendpoint\trequirement_ns\tslack_ns"
set trace_failed 0
foreach group {request response command result} expected {5 10 5 10} {
    set source [trace_pins [set trace_${group}_source] C]
    set target [trace_pins [set trace_${group}_target] D]
    set clocks [get_clocks -of_objects [trace_pins [set trace_${group}_target] C]]
    if {[llength $clocks] != 1 || abs([get_property PERIOD $clocks] - $expected) > 0.001} {
        error "NPU trace destination clock differs for $group: $clocks"
    }
    set_max_delay -datapath_only $expected -from $source -to $target
    set paths [get_timing_paths -delay_type max -from $source -to $target -nworst 1 -max_paths [llength $target]]
    set visited [dict create]
    foreach path $paths {
        set endpoint [get_property ENDPOINT_PIN $path]
        set requirement [get_property REQUIREMENT $path]
        set slack [get_property SLACK $path]
        dict set visited $endpoint 1
        puts $trace_report "$group\t$endpoint\t$requirement\t$slack"
        if {abs($requirement-$expected) > 0.001 || ($trace_check_slack && $slack < 0)} {set trace_failed 1}
    }
    foreach pin $target {
        if {![dict exists $visited [get_property NAME $pin]]} {
            puts $trace_report "$group\t$pin\tMISSING\tMISSING"
            set trace_failed 1
        }
    }
}
close $trace_report
if {$trace_failed} {error "NPU trace CDC coverage or timing failed; reports retained"}
