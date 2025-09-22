/**
 * 结束节点组件
 * 对话流的终点
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { DialogNodeData } from '../VisualDialogEditor';

interface EndNodeProps extends NodeProps {
  data: DialogNodeData;
}

const EndNode: React.FC<EndNodeProps> = ({ data, selected }) => {
  return (
    <div className={`
      bg-white border-2 rounded-full shadow-lg w-24 h-24 flex items-center justify-center
      ${selected ? 'border-red-600' : 'border-red-400'}
      hover:border-red-500 transition-colors
    `}>
      {/* 输入连接点 */}
      <Handle
        type="target"
        position={Position.Top}
        className="w-4 h-4 bg-red-500 border-2 border-white"
      />
      
      {/* 节点内容 */}
      <div className="text-center">
        <div className="bg-red-500 rounded-full w-16 h-16 flex items-center justify-center">
          <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 012 0v4a1 1 0 11-2 0V7zM12 7a1 1 0 012 0v4a1 1 0 11-2 0V7z" clipRule="evenodd" />
          </svg>
        </div>
      </div>
      
      {/* 标签 */}
      <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2">
        <span className="text-xs font-medium text-red-600 bg-white px-2 py-1 rounded shadow">
          END
        </span>
      </div>
    </div>
  );
};

export default EndNode;
