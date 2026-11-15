# 所有新跨域都保留数据路径预算；旧厂商例外在准备阶段排除 yolo26_video。
source [file join [file dirname [info script]] y26_control_cdc.tcl]
set y26_link_source [y26_cdc_registers {(^|.*/)yolo26_video/link_status_hold_reg\[[0-9]+\]}]
set y26_link_target [y26_cdc_registers {(^|.*/)yolo26_video/link_status_sync0_reg\[[0-9]+\]}]
if {[llength $y26_link_source] != 6 || [llength $y26_link_target] != 6} {
    error "YOLO26 link status synchronizer width changed"
}
y26_cdc_bound link_status $y26_link_source $y26_link_target 10.000
