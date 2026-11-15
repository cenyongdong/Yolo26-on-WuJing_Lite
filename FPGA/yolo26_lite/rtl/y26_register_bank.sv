`timescale 1ns / 1ps

// GP 时钟域的寄存器文件。命令快照由本模块持有，直到软件确认对应的完整响应。
// 写入在一个周期内接收；非法或忙时写入只记粘滞错误，不覆盖在途参数。
// STOP 是独立电平通道，不能被等待 DMA 完成的命令邮箱挡住。
module y26_register_bank (
    input wire clk,
    input wire resetn,
    input wire write_valid,
    output wire write_ready,
    input wire [7:0] write_offset,
    input wire [31:0] write_data,
    input wire [3:0] write_strobe,
    input wire read_valid,
    output wire read_ready,
    input wire [7:0] read_offset,
    output reg read_response_valid,
    input wire read_response_ready,
    output reg [31:0] read_response_data,

    output reg command_valid,
    input wire command_ready,
    output reg [383:0] command_payload,
    input wire mailbox_busy,
    input wire response_valid,
    output wire response_ready,
    input wire [223:0] response_payload,
    output reg stop_requested
);
    localparam [7:0] CSR_ABI = 8'h00, CSR_COMMAND = 8'h04, CSR_STATE = 8'h08,
                     CSR_ERRORS = 8'h0c, CSR_TAG = 8'h10, CSR_MAP_DATA = 8'h38,
                     CSR_STOP = 8'h3c, CSR_RESPONSE = 8'h40, CSR_ACK = 8'h5c,
                     CSR_ISSUED = 8'h60, CSR_COMPLETED = 8'h64;
    localparam [31:0] ABI_VERSION = 32'h59323601;
    localparam integer ERROR_ADDRESS = 0, ERROR_BUSY = 1, ERROR_STROBE = 2,
                       ERROR_OPCODE = 3, ERROR_ACK = 4, ERROR_STOP_RELEASE = 5,
                       ERROR_RESPONSE = 6;
    reg [31:0] parameters [0:10];
    reg [223:0] snapshot;
    reg snapshot_valid;
    reg [31:0] errors, issued_count, completed_count;
    wire locked = command_valid || mailbox_busy || snapshot_valid;
    wire write_fire = write_valid && write_ready;
    wire read_fire = read_valid && read_ready;
    wire parameter_write = write_offset >= CSR_TAG && write_offset <= CSR_MAP_DATA;
    wire unsafe_response = !response_payload[160] ||
                           response_payload[63:32] != command_payload[63:32] ||
                           response_payload[223:192] != command_payload[31:0];
    assign write_ready = resetn;
    assign read_ready = resetn && (!read_response_valid || read_response_ready);
    assign response_ready = resetn && !snapshot_valid;

    integer index, byte_index;
    reg [31:0] selected_read;
    always @* begin
        selected_read = 0;
        case (read_offset)
            CSR_ABI: selected_read = ABI_VERSION;
            CSR_STATE: selected_read = {26'd0, (errors != 0), stop_requested, snapshot_valid,
                                       mailbox_busy, command_valid, locked};
            CSR_ERRORS: selected_read = errors;
            CSR_STOP: selected_read = {31'd0, stop_requested};
            CSR_ISSUED: selected_read = issued_count;
            CSR_COMPLETED: selected_read = completed_count;
            default: begin
                for (integer word_index = 0; word_index < 11; word_index = word_index + 1) begin
                    if (read_offset == CSR_TAG + word_index * 4) begin
                        selected_read = parameters[word_index];
                    end
                end
                for (integer word_index = 0; word_index < 7; word_index = word_index + 1) begin
                    if (read_offset == CSR_RESPONSE + word_index * 4) begin
                        selected_read = snapshot_valid ? snapshot[word_index * 32 +: 32] : 32'd0;
                    end
                end
            end
        endcase
    end

    always @(posedge clk) begin
        if (!resetn) begin
            command_valid <= 0;
            command_payload <= 0;
            stop_requested <= 0;
            snapshot_valid <= 0;
            snapshot <= 0;
            errors <= 0;
            issued_count <= 0;
            completed_count <= 0;
            read_response_valid <= 0;
            read_response_data <= 0;
            for (index = 0; index < 11; index = index + 1) begin
                parameters[index] <= 0;
            end
        end else begin
            if (read_response_valid && read_response_ready) begin
                read_response_valid <= 0;
            end
            if (read_fire) begin
                read_response_valid <= 1;
                read_response_data <= selected_read;
            end
            if (command_valid && command_ready) begin
                command_valid <= 0;
            end
            if (response_valid && response_ready) begin
                snapshot <= response_payload;
                snapshot_valid <= 1;
                completed_count <= completed_count + 1;
            end
            if (write_fire) begin
                if (write_offset[1:0] != 0) begin
                    errors[ERROR_ADDRESS] <= 1;
                end else if (parameter_write) begin
                    if (locked) begin
                        errors[ERROR_BUSY] <= 1;
                    end else begin
                        for (index = 0; index < 11; index = index + 1) begin
                            if (write_offset == CSR_TAG + index * 4) begin
                                for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1) begin
                                    if (write_strobe[byte_index]) begin
                                        parameters[index][byte_index * 8 +: 8] <= write_data[byte_index * 8 +: 8];
                                    end
                                end
                            end
                        end
                    end
                end else if (write_strobe != 4'hf) begin
                    errors[ERROR_STROBE] <= 1;
                end else begin
                    case (write_offset)
                        CSR_COMMAND: begin
                            if (locked) begin
                                errors[ERROR_BUSY] <= 1;
                            end else if (write_data < 1 || write_data > 3) begin
                                errors[ERROR_OPCODE] <= 1;
                            end else begin
                                command_payload <= {parameters[10], parameters[9], parameters[8],
                                                    parameters[7], parameters[6], parameters[5],
                                                    parameters[4], parameters[3], parameters[2],
                                                    parameters[1], parameters[0], write_data};
                                command_valid <= 1;
                                issued_count <= issued_count + 1;
                            end
                        end
                        CSR_ERRORS: begin
                            errors <= errors & ~write_data;
                            // 响应安全错误只允许协调复位后清除；清状态位不等于恢复了所有权。
                            errors[ERROR_RESPONSE] <= errors[ERROR_RESPONSE];
                        end
                        CSR_STOP: begin
                            if (write_data == 1) begin
                                stop_requested <= 1;
                            end else if (write_data != 0) begin
                                errors[ERROR_OPCODE] <= 1;
                            end else if (locked) begin
                                errors[ERROR_STOP_RELEASE] <= 1;
                            end else begin
                                stop_requested <= 0;
                            end
                        end
                        CSR_ACK: begin
                            if (snapshot_valid && write_data == snapshot[63:32] && snapshot[160] &&
                                snapshot[63:32] == command_payload[63:32] &&
                                snapshot[223:192] == command_payload[31:0] && !errors[ERROR_RESPONSE]) begin
                                snapshot_valid <= 0;
                            end else begin
                                errors[ERROR_ACK] <= 1;
                            end
                        end
                        default: errors[ERROR_ADDRESS] <= 1;
                    endcase
                end
            end
            // 硬件报告优先于同拍软件 W1C；清状态位不等于恢复了对象所有权。
            if (response_valid && response_ready && unsafe_response) begin
                errors[ERROR_RESPONSE] <= 1;
            end
        end
    end
endmodule
