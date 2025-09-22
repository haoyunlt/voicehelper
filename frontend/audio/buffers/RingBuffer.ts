/**
 * Ring Buffer - 环形缓冲区
 * 功能: 高效的固定大小循环缓冲区，用于音频数据流处理
 */

export class RingBuffer {
    private buffer: Float32Array;
    private writeIndex: number = 0;
    private readIndex: number = 0;
    private size: number;
    private capacity: number;
    private overflowCount: number = 0;
    private underflowCount: number = 0;
    
    constructor(capacity: number) {
        this.capacity = capacity;
        this.buffer = new Float32Array(capacity);
        this.size = 0;
    }
    
    /**
     * 写入数据到缓冲区
     */
    write(data: Float32Array): number {
        const availableSpace = this.capacity - this.size;
        const writeLength = Math.min(data.length, availableSpace);
        
        if (writeLength < data.length) {
            this.overflowCount++;
        }
        
        for (let i = 0; i < writeLength; i++) {
            this.buffer[this.writeIndex] = data[i];
            this.writeIndex = (this.writeIndex + 1) % this.capacity;
        }
        
        this.size += writeLength;
        return writeLength;
    }
    
    /**
     * 从缓冲区读取数据
     */
    read(output: Float32Array): number {
        const readLength = Math.min(output.length, this.size);
        
        if (readLength < output.length) {
            this.underflowCount++;
            // 填充剩余部分为0
            output.fill(0, readLength);
        }
        
        for (let i = 0; i < readLength; i++) {
            output[i] = this.buffer[this.readIndex];
            this.readIndex = (this.readIndex + 1) % this.capacity;
        }
        
        this.size -= readLength;
        return readLength;
    }
    
    /**
     * 强制写入数据，覆盖旧数据
     */
    forceWrite(data: Float32Array): void {
        for (let i = 0; i < data.length; i++) {
            this.buffer[this.writeIndex] = data[i];
            this.writeIndex = (this.writeIndex + 1) % this.capacity;
            
            // 如果缓冲区满了，移动读指针
            if (this.size >= this.capacity) {
                this.readIndex = (this.readIndex + 1) % this.capacity;
                this.overflowCount++;
            } else {
                this.size++;
            }
        }
    }
    
    /**
     * 预览数据但不移动读指针
     */
    peek(output: Float32Array, offset: number = 0): number {
        const availableData = this.size - offset;
        const peekLength = Math.min(output.length, Math.max(0, availableData));
        
        let peekIndex = (this.readIndex + offset) % this.capacity;
        
        for (let i = 0; i < peekLength; i++) {
            output[i] = this.buffer[peekIndex];
            peekIndex = (peekIndex + 1) % this.capacity;
        }
        
        // 填充剩余部分为0
        if (peekLength < output.length) {
            output.fill(0, peekLength);
        }
        
        return peekLength;
    }
    
    /**
     * 跳过指定数量的样本
     */
    skip(count: number): number {
        const skipLength = Math.min(count, this.size);
        this.readIndex = (this.readIndex + skipLength) % this.capacity;
        this.size -= skipLength;
        return skipLength;
    }
    
    /**
     * 获取可用数据量
     */
    getAvailableData(): number {
        return this.size;
    }
    
    /**
     * 获取可用空间
     */
    getAvailableSpace(): number {
        return this.capacity - this.size;
    }
    
    /**
     * 检查是否为空
     */
    isEmpty(): boolean {
        return this.size === 0;
    }
    
    /**
     * 检查是否已满
     */
    isFull(): boolean {
        return this.size >= this.capacity;
    }
    
    /**
     * 清空缓冲区
     */
    clear(): void {
        this.writeIndex = 0;
        this.readIndex = 0;
        this.size = 0;
    }
    
    /**
     * 获取填充率 (0-1)
     */
    getFillRatio(): number {
        return this.size / this.capacity;
    }
    
    /**
     * 获取统计信息
     */
    getStats() {
        return {
            capacity: this.capacity,
            size: this.size,
            fillRatio: this.getFillRatio(),
            overflowCount: this.overflowCount,
            underflowCount: this.underflowCount,
            availableSpace: this.getAvailableSpace()
        };
    }
    
    /**
     * 重置统计计数器
     */
    resetStats(): void {
        this.overflowCount = 0;
        this.underflowCount = 0;
    }
    
    /**
     * 调整缓冲区大小
     */
    resize(newCapacity: number): void {
        const oldBuffer = new Float32Array(this.size);
        this.read(oldBuffer);
        
        this.capacity = newCapacity;
        this.buffer = new Float32Array(newCapacity);
        this.writeIndex = 0;
        this.readIndex = 0;
        this.size = 0;
        
        // 重新写入数据
        this.write(oldBuffer);
    }
}
