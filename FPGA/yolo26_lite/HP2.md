# HP2 共享入口与仲裁

## 已核对的原系统

原 `ps_ai_peripheral.sv` 保留 AI 四路，HC2 接 HP2 互联 S00，显示读与原相机写接 S01。
原厂生成 `ps_ai_xbar_4` 的数据宽为 64，AXI3、PS ID 宽 6、S00/S01 原始 ID 宽各 5，
`C_S_AXI_BASE_ID=0x0000002000000000`。S01 原始 ID 0 在 PS 侧为 0x20，返回再恢复为 0。
不能把 0x20 自身当作错误，也不能直接截断 LEN 或 ID。

本机参考位置：`D:/FPGACompetitionProject/.local/hdmi-boot-witness-20261019-r23/reference`。
已直接核对其中 `fpai_demo_vivado.srcs/sources_1/bd/ps_ai/ip/ps_ai_xbar_4/synth/ps_ai_xbar_4.v`、
`rtl/display/frame_to_ps_largebuf.v` 和 `rtl/display/aq_axi_master.v`。
`CANDIDATE_SCANOUT=1` 关闭的是旧显示读取和旧发送时钟，**并未删除旧相机写主机**；
原写主机 AWID 固定 0，既有 WID 也为 0。因此新 S2MM 不能直接覆盖该通道。

## 新模块

`y26_axi_read_share` 将 HDMI（客户 0）与预处理 MM2S（客户 1）组合到原 S01 读入口。
一次拥有一个完整突发，接收客户 AR 后锁存全部地址属性，再向原互联发送。
显示 FIFO 低水位时可连续优先最多四个突发；预处理也持续等待时，第五次必定服务它。
非紧急时轮询。优先级只影响尚未接受的请求，不改变已经呈现的 AR。

`y26_axi_write_share` 将原相机写（客户 0）与预处理 S2MM（客户 1）组合到原 S01 写入口。
一次拥有一个 AW、其全部 W 和对应 B，按客户轮询。上游可提前呈现 W，但本模块在地址握手后才接受数据。
原 ID／地址属性保持；新 YOLO 读写计划使用原始 ID 1，PS 侧映射为 0x21，仍使用既有 S01 5 位空间。
返回按已接受的事务所有者交付，并检查完整 RID/BID/WID。

错误 ID 不投递给任何帧；错误 WLAST 不向总线提交。故障保留在途状态和独占，不自动恢复。
普通 RRESP/BRESP 错误仍向真实所有者传递并按协议排空，由原 reader／DMA 返回失败。
返回计数不能代替 RLAST／B 完成，模块没有“超时后变空闲”路径。
启动前的未知读返回留给现有有界 startup drain，仲裁器不擅自消费；功能请求与未知返回并存时拒绝启动。

读写仲裁各独立，彼此可以并行；HC2 及另外三路 AI 的连接、ID 和原厂互联不变。
当前只是可复用 RTL，尚未接入 `ps_ai_peripheral`，显示低水位、故障寄存器和原启动排空的联合接线仍待完成。

## 验证

- 读侧 257 项：两客户各 128 个 1..16 拍突发、全部地址属性、优先级改变、随机地址／返回背压、
  轮询与最多四次显示优先、响应错误透传、启动未知返回、错误 RID 保留独占。
  首轮 TB 在末拍同一 delta cycle 检查组合 idle，修正采样等待后通过，原失败记录保留。
  见 [read-share-r1.json](evidence/read-share-r1.json)。
- 写侧 260 项：两客户各 128 个突发，W 先呈现、AW／W／B 背压、非连续 WSTRB、完整 5bit ID、
  响应错误透传，以及错误 WID、早/晚 WLAST、错误 BID 四类保护。
  见 [write-share-r1.json](evidence/write-share-r1.json)。

以上为原厂 XSim 2018.3 功能验证，不是厂商交叉开关联合仿真、整机布局布线或 DDR 板测。
