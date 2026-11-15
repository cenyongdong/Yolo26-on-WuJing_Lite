# 共享平台既有定点时序例外

## 特批范围

例外标签“已知原厂安全占位时钟”作为工程例外标签记录；“原厂bug”“安全”“与核心功能无关”及占位语义尚未独立证实。已经证实的是：原厂AI_Mate网表包含该时钟模型，路由检查点可见叶负载为ai_core_clk_locked_reg/C。在该既定例外范围内，可以继续受控板测，无需再次批准这两个检查。

共同路径前缀为`U0_ps_ai_peripheral/ZGolo/u_AI_Mate_top/clkrst_wrapper_u0/AI_CORE_CLK/inst/`，共同为`clk_out1_AI_CORE_CLK_ST`、报告周期1.000ns。

| 定点对象（前缀后） | 检查 | 要求 | 原裕量 | 放行方式 |
|---|---|---|---|---|
| clkout1_buf/I | Min Period | 1.409ns | −0.409ns | 定点例外 |
| plle2_adv_inst/CLKOUT0 | Min Period | 1.071ns | −0.071ns | 定点例外 |

WPWS−0.409ns、TPWS−0.479ns和两个失败端点按原报告保存；不改库、频率、约束或AI_Mate，不通过设置全域false_path包装通过。没有新增其它时序/DRC豁免。CDC及外部时序未完整签核须在实测结果中继续披露。


YOLO26l 实际签核见 [物理证据](../yolo26_lite/evidence/rectangular-video-physical.json)。
