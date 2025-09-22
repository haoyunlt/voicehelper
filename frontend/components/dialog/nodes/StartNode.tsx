/**
 * 开始节点组件
 * 对话流的入口点
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { DialogNodeData } from '../VisualDialogEditor';

interface StartNodeProps extends NodeProps {
  data: DialogNodeData;
}

const StartNode: React.FC<StartNodeProps> = ({ data, selected }) => {
  return (
    <div className={`
      bg-white border-2 rounded-full shadow-lg w-24 h-24 flex items-center justify-center
      ${selected ? 'border-green-600' : 'border-green-400'}
      hover:border-green-500 transition-colors
    `}>
      {/* 节点内容 */}
      <div className="text-center">
        <div className="bg-green-500 rounded-full w-16 h-16 flex items-center justify-center">
          <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
          </svg>
        </div>
      </div>
      
      {/* 输出连接点 */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="w-4 h-4 bg-green-500 border-2 border-white"
      />
      
      {/* 标签 */}
      <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2">
        <span className="text-xs font-medium text-green-600 bg-white px-2 py-1 rounded shadow">
          START
        </span>
      </div>
    </div>
  );
};

export default StartNode;
