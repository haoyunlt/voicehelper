/**
 * 意图节点组件
 * 用于识别用户意图的节点
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { DialogNodeData } from '../VisualDialogEditor';

interface IntentNodeProps extends NodeProps {
  data: DialogNodeData;
}

const IntentNode: React.FC<IntentNodeProps> = ({ data, selected }) => {
  return (
    <div className={`
      bg-white border-2 rounded-lg shadow-lg min-w-[200px] max-w-[300px]
      ${selected ? 'border-blue-500' : 'border-gray-300'}
      hover:border-blue-400 transition-colors
    `}>
      {/* 输入连接点 */}
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 bg-blue-500 border-2 border-white"
      />
      
      {/* 节点头部 */}
      <div className="bg-blue-500 text-white px-3 py-2 rounded-t-lg flex items-center">
        <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" clipRule="evenodd" />
        </svg>
        <span className="text-sm font-medium">Intent</span>
      </div>
      
      {/* 节点内容 */}
      <div className="p-3">
        <div className="mb-2">
          <div className="text-sm font-medium text-gray-700 mb-1">
            {data.label || 'Untitled Intent'}
          </div>
          <div className="text-xs text-gray-500 bg-gray-50 p-2 rounded border">
            {data.content || 'No intent pattern defined'}
          </div>
        </div>
        
        {/* 条件列表 */}
        {data.conditions && data.conditions.length > 0 && (
          <div className="mt-2">
            <div className="text-xs font-medium text-gray-600 mb-1">Patterns:</div>
            <div className="space-y-1">
              {data.conditions.slice(0, 3).map((condition, index) => (
                <div key={index} className="text-xs text-gray-500 bg-blue-50 px-2 py-1 rounded">
                  {condition || `Pattern ${index + 1}`}
                </div>
              ))}
              {data.conditions.length > 3 && (
                <div className="text-xs text-gray-400">
                  +{data.conditions.length - 3} more patterns
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* 置信度指示器 */}
        <div className="mt-2 flex items-center justify-between">
          <span className="text-xs text-gray-500">Confidence</span>
          <div className="flex space-x-1">
            {[1, 2, 3, 4, 5].map((level) => (
              <div
                key={level}
                className={`w-2 h-2 rounded-full ${
                  level <= (data.metadata?.confidence || 3)
                    ? 'bg-green-400'
                    : 'bg-gray-200'
                }`}
              />
            ))}
          </div>
        </div>
      </div>
      
      {/* 输出连接点 */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 bg-blue-500 border-2 border-white"
      />
    </div>
  );
};

export default IntentNode;
