/**
 * 响应节点组件
 * 用于定义机器人回复的节点
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { DialogNodeData } from '../VisualDialogEditor';

interface ResponseNodeProps extends NodeProps {
  data: DialogNodeData;
}

const ResponseNode: React.FC<ResponseNodeProps> = ({ data, selected }) => {
  return (
    <div className={`
      bg-white border-2 rounded-lg shadow-lg min-w-[200px] max-w-[300px]
      ${selected ? 'border-green-500' : 'border-gray-300'}
      hover:border-green-400 transition-colors
    `}>
      {/* 输入连接点 */}
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 bg-green-500 border-2 border-white"
      />
      
      {/* 节点头部 */}
      <div className="bg-green-500 text-white px-3 py-2 rounded-t-lg flex items-center">
        <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M18 13V5a2 2 0 00-2-2H4a2 2 0 00-2 2v8a2 2 0 002 2h3l3 3 3-3h3a2 2 0 002-2zM5 7a1 1 0 011-1h8a1 1 0 110 2H6a1 1 0 01-1-1zm1 3a1 1 0 100 2h3a1 1 0 100-2H6z" clipRule="evenodd" />
        </svg>
        <span className="text-sm font-medium">Response</span>
      </div>
      
      {/* 节点内容 */}
      <div className="p-3">
        <div className="mb-2">
          <div className="text-sm font-medium text-gray-700 mb-1">
            {data.label || 'Untitled Response'}
          </div>
          <div className="text-xs text-gray-600 bg-gray-50 p-2 rounded border min-h-[40px]">
            {data.content || 'No response content defined'}
          </div>
        </div>
        
        {/* 响应变体 */}
        {data.responses && data.responses.length > 0 && (
          <div className="mt-2">
            <div className="text-xs font-medium text-gray-600 mb-1">
              Variants ({data.responses.length}):
            </div>
            <div className="space-y-1">
              {data.responses.slice(0, 2).map((response, index) => (
                <div key={index} className="text-xs text-gray-500 bg-green-50 px-2 py-1 rounded">
                  {response.length > 50 ? `${response.substring(0, 50)}...` : response}
                </div>
              ))}
              {data.responses.length > 2 && (
                <div className="text-xs text-gray-400">
                  +{data.responses.length - 2} more variants
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* 响应类型和设置 */}
        <div className="mt-2 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {/* 响应类型图标 */}
            {data.metadata?.responseType === 'voice' && (
              <div className="flex items-center text-xs text-blue-600">
                <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.617.793L4.617 14H2a1 1 0 01-1-1V7a1 1 0 011-1h2.617l3.766-2.793a1 1 0 011.617.793z" clipRule="evenodd" />
                </svg>
                Voice
              </div>
            )}
            
            {data.metadata?.hasQuickReplies && (
              <div className="flex items-center text-xs text-purple-600">
                <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 8a6 6 0 01-7.743 5.743L10 14l-1 1-1 1H4a2 2 0 01-2-2v-5L4.257 6.743A6 6 0 0118 8z" clipRule="evenodd" />
                </svg>
                Quick
              </div>
            )}
            
            {data.metadata?.hasCards && (
              <div className="flex items-center text-xs text-orange-600">
                <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
                </svg>
                Cards
              </div>
            )}
          </div>
          
          {/* 延迟指示器 */}
          {data.metadata?.delay && (
            <div className="text-xs text-gray-500">
              {data.metadata.delay}ms
            </div>
          )}
        </div>
      </div>
      
      {/* 输出连接点 */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 bg-green-500 border-2 border-white"
      />
    </div>
  );
};

export default ResponseNode;
