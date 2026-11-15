# 综合/链接后的 y26_control 专用约束。保持数据先于三拍通知到达；不切断跨域时序路径。
# 仅约束本外设的实际寄存器，不能对整个 GP/HP 时钟对添加例外。
proc y26_cdc_registers {pattern} {
    set cells [get_cells -hier -regexp $pattern]
    if {[llength $cells] == 0} {error "YOLO26 CDC register group missing: $pattern"}
    return $cells
}
proc y26_cdc_pins {cells pin_name} {
    set pins [get_pins -of_objects $cells -filter "REF_PIN_NAME == $pin_name"]
    if {[llength $pins] != [llength $cells]} {error "YOLO26 CDC register pin coverage: $pin_name"}
    return $pins
}
proc y26_cdc_bound {label source_cells target_cells expected_period} {
    set source_clock_pins [y26_cdc_pins $source_cells C]
    set target_data_pins [y26_cdc_pins $target_cells D]
    set capture [get_clocks -of_objects [y26_cdc_pins $target_cells C]]
    if {[llength $capture] != 1} {error "YOLO26 CDC capture clock missing or ambiguous: $label"}
    set period [get_property PERIOD $capture]
    if {abs($period - $expected_period) > 0.001} {
        error "YOLO26 CDC unexpected period $period for $label; expected $expected_period"
    }
    # C 引脚是有效时序起点。读取已布线报告时只查询端点，使用 DCP 已保存的约束。
    if {![info exists ::y26_cdc_report_only]} {
        set_max_delay -datapath_only $period -from $source_clock_pins -to $target_data_pins
    }
    puts "YOLO26_CDC $label source=[llength $source_cells] target=[llength $target_cells] max_ns=$period"
}

set y26_request_source [y26_cdc_registers {(^|.*/)control/mailbox/request_toggle_reg}]
set y26_request_target [y26_cdc_registers {(^|.*/)control/mailbox/request_sync_reg\[0\]}]
set y26_response_source [y26_cdc_registers {(^|.*/)control/mailbox/response_toggle_reg}]
set y26_response_target [y26_cdc_registers {(^|.*/)control/mailbox/response_sync_reg\[0\]}]
set y26_command_source [y26_cdc_registers {(^|.*/)control/mailbox/payload_hold_reg\[[0-9]+\]}]
set y26_command_target [y26_cdc_registers {(^|.*/)control/mailbox/target_command_reg\[[0-9]+\]}]
set y26_result_source [y26_cdc_registers {(^|.*/)control/mailbox/response_hold_reg\[[0-9]+\]}]
set y26_result_target [y26_cdc_registers {(^|.*/)control/mailbox/source_response_reg\[[0-9]+\]}]
set y26_stop_source [y26_cdc_registers {(^|.*/)control/registers/stop_requested_reg}]
set y26_stop_target [y26_cdc_registers {(^|.*/)control/stop_sync_reg\[0\]}]
y26_cdc_bound request $y26_request_source $y26_request_target 5.000
y26_cdc_bound response $y26_response_source $y26_response_target 10.000
y26_cdc_bound command_payload $y26_command_source $y26_command_target 5.000
y26_cdc_bound response_payload $y26_result_source $y26_result_target 10.000
y26_cdc_bound stop $y26_stop_source $y26_stop_target 5.000
