`timescale 1ns/1ps
module scanout_tests #(
 parameter integer HS=44, HB=148, HA=1920, HF=88,
 parameter integer VS=5, VB=36, VA=1080, VF=4,
 parameter real PIXEL_HALF_NS=3.367
)(output reg complete=0);
 localparam integer FRAME_PIXELS=HA*VA, FRAME_WORDS=FRAME_PIXELS/4;
 localparam [31:0] BASE=32'h30100fc0;
 reg csr=0,mem=0,pix=0,resetn=0,pixel_clock_enabled=1;
 reg wvalid=0;wire wready;reg [7:0] offset=0,read_offset=0;reg [31:0] value=0;
 wire [31:0] read_data,araddr;wire [7:0] arlen;wire [2:0] arsize;wire [1:0] arburst;
 wire arvalid,rready,idle,hs,vs,de,read_urgent;wire [23:0] rgb;
 reg saw_low_water=0,saw_high_water=0;
 reg active=0;reg [31:0] addr=0;reg [7:0] index=0,last=0;
 reg supply=1,allow_address=1;
 integer requests=0,words=0,pixels=0,lines=0,hs_width=0,de_width=0,vs_lines=0;
 integer old_hs=-1,old_vs=-1,phase=0,issued=0,pixel_cycles=0;
 reg prior_hs=0,prior_vs=0,capturing=0;
 always #7 csr=~csr;
 always #2.5 mem=~mem;
 always begin #(PIXEL_HALF_NS);if(pixel_clock_enabled)pix=~pix;end
 function [15:0] pattern(input integer n);pattern=(n*17)^16'habcd;endfunction
 function [23:0] expand(input [15:0] p);expand={p[15:11],p[15:13],p[10:5],p[10:9],p[4:0],p[4:2]};endfunction
 function [63:0] frame_word(input [31:0] a);
   integer n;begin n=(a-BASE)/2;frame_word={pattern(n+3),pattern(n+2),pattern(n+1),pattern(n)};end
 endfunction
 hdmi_scanout #(.FIFO_AW(9),.HS(HS),.HB(HB),.HA(HA),.HF(HF),.VS(VS),.VB(VB),.VA(VA),.VF(VF)) dut(
   .csr_clk(csr),.csr_resetn(resetn),.write_valid(wvalid),.write_ready(wready),
   .write_offset(offset),.write_data(value),.write_strobe(4'hf),.read_offset(read_offset),.read_data(read_data),
   .mem_clk(mem),.mem_resetn(resetn),.pixel_clk(pix),.pixel_resetn(resetn),.clock_locked(1'b1),
   .araddr(araddr),.arlen(arlen),.arsize(arsize),.arburst(arburst),.arvalid(arvalid),.arready(allow_address&&!active),
   .rdata(frame_word(addr+index*8)),.rresp(2'b0),.rlast(index==last),.rvalid(active&&supply),.rready(rready),
   .safe_idle(idle),.read_urgent(read_urgent),.hs(hs),.vs(vs),.de(de),.rgb(rgb));
 always @(posedge mem) if(resetn) begin
   if(read_urgent)saw_low_water<=1;else saw_high_water<=1;
   if(arvalid && allow_address && !active) begin
     if(arsize!=3 || arburst!=1 || arlen>15 || (araddr[11:0]+(arlen+1)*8)>4096)
       $fatal(1,"Scanout illegal AXI request");
     active<=1;addr<=araddr;last<=arlen;index<=0;requests=requests+1;
   end
   if(active && supply && rready) begin
     words=words+1;if(index==last)active<=0;else index<=index+1'b1;
   end
   if(idle && (active || arvalid)) $fatal(1,"Scanout false idle");
 end
 // External raster/data observations; counters check the actual registered pins.
 always @(negedge pix) if(resetn) begin
   pixel_cycles=pixel_cycles+1;
   if(hs) hs_width=hs_width+1;
   if(de) de_width=de_width+1;
   if(hs && !prior_hs) begin
     if(old_hs>=0 && pixel_cycles-old_hs!=HS+HB+HA+HF)$fatal(1,"Horizontal total changed");
     old_hs=pixel_cycles;
     if(lines>0 && hs_width!=1+HS) $fatal(1,"HS width changed: %0d",hs_width);
     hs_width=1;lines=lines+1;
     if(de_width!=0 && de_width!=HA) $fatal(1,"Active line width changed");
     de_width=0;
     if(vs) vs_lines=vs_lines+1;
   end
   if(vs && !prior_vs) begin
     if(old_vs>=0 && lines-old_vs!=VS+VB+VA+VF) $fatal(1,"Vertical total changed");
     if(old_vs>=0 && vs_lines!=VS+1) $fatal(1,"VS line width changed %0d",vs_lines);
     old_vs=lines;vs_lines=1;
   end
   prior_hs=hs;prior_vs=vs;
   if(phase==1 && dut.frame_toggle) capturing=1;
   if(phase==1 && capturing && de) begin
     if(rgb!==expand(pattern(pixels))) $fatal(1,"Scanout pixel mismatch n=%0d rgb=%h expected=%h",pixels,rgb,expand(pattern(pixels)));
     pixels=pixels+1;
   end
   if(phase==2 && rgb!==0) $fatal(1,"Stopped raster must be black");
 end
 task write(input [7:0] a,input [31:0] d);
 begin @(negedge csr);offset=a;value=d;wvalid=1;do @(posedge csr);while(!wready);@(negedge csr);wvalid=0;end endtask
 task command(input [31:0] op,input [31:0] id);
 begin
   write(8'h08,id);write(8'h0c,op);issued=issued+1;
   read_offset=8'h28;wait(read_data==issued);@(negedge csr);
 end endtask
 initial begin
   #101;resetn=1;repeat(15) @(negedge csr);
   write(8'h04,BASE);phase=1;command(1,101);
   wait(pixels==FRAME_PIXELS);@(negedge csr);command(3,102);wait(idle);
   repeat(10) @(negedge pix);phase=2;
   if(words!=FRAME_WORDS || requests!=FRAME_WORDS/16+1 || dut.underflow || dut.core.core.control.faults!=0 ||
      !saw_low_water || !saw_high_water)
      $fatal(1,"Complete scanout count/fault failure");
   command(0,0);read_offset=8'h14;#1;if(read_data!=102)$fatal(1,"Stop tag not observed");
   // Restart with no data supplied: the first missing active pixel faults.
   supply=0;phase=3;command(1,103);wait(active);wait(dut.underflow);
   repeat(15) @(negedge mem);
   if(idle || !dut.core.core.control.faults[3])$fatal(1,"Underflow must stop but await accepted AXI drain");
   supply=1;wait(idle);repeat(15) @(negedge csr);command(4,0);
   command(0,0);read_offset=8'h20;#1;if(read_data!=0)$fatal(1,"Underflow clear failed");
   // Accepted burst can be safely drained even with pixel clock halted.
   command(1,104);wait(active);pixel_clock_enabled=0;command(3,105);wait(idle);
   pixel_clock_enabled=1;repeat(20) @(negedge pix);phase=2;
   repeat(2300) @(negedge pix);
   $display("CANDIDATE_SCANOUT_PASS pixels=%0d bytes=%0d base_offset=4032 raster=%0dx%0d hs=%0d vs=%0d underflow_drain=1 stop_pixel_clock_halted=1",
            FRAME_PIXELS,FRAME_PIXELS*2,HS+HB+HA+HF,VS+VB+VA+VF,HS,VS);
   complete=1;
 end
 initial begin #90000000;if(!complete)$fatal(1,"Scanout watchdog phase=%0d pixels=%0d",phase,pixels);end
endmodule
