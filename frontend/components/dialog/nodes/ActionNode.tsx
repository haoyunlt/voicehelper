/**
 * 动作节点组件
 * 用于执行特定操作的节点
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { DialogNodeData } from '../VisualDialogEditor';

interface ActionNodeProps extends NodeProps {
  data: DialogNodeData;
}

const ActionNode: React.FC<ActionNodeProps> = ({ data, selected }) => {
  return (
    <div className={`
      bg-white border-2 rounded-lg shadow-lg min-w-[200px] max-w-[300px]
      ${selected ? 'border-purple-500' : 'border-gray-300'}
      hover:border-purple-400 transition-colors
    `}>
      {/* 输入连接点 */}
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 bg-purple-500 border-2 border-white"
      />
      
      {/* 节点头部 */}
      <div className="bg-purple-500 text-white px-3 py-2 rounded-t-lg flex items-center">
        <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clipRule="evenodd" />
        </svg>
        <span className="text-sm font-medium">Action</span>
      </div>
      
      {/* 节点内容 */}
      <div className="p-3">
        <div className="mb-2">
          <div className="text-sm font-medium text-gray-700 mb-1">
            {data.label || 'Untitled Action'}
          </div>
          <div className="text-xs text-gray-600 bg-gray-50 p-2 rounded border">
            {data.content || 'No action defined'}
          </div>
        </div>
        
        {/* 动作列表 */}
        {data.actions && data.actions.length > 0 && (
          <div className="mt-2">
            <div className="text-xs font-medium text-gray-600 mb-1">Operations:</div>
            <div className="space-y-1">
              {data.actions.slice(0, 3).map((action, index) => (
                <div key={index} className="flex items-center justify-between bg-purple-50 px-2 py-1 rounded">
                  <span className="text-xs text-gray-600">
                    {action.length > 25 ? `${action.substring(0, 25)}...` : action}
                  </span>
                  <span className="text-xs text-purple-600 font-medium">
                    #{index + 1}
                  </span>
                </div>
              ))}
              {data.actions.length > 3 && (
                <div className="text-xs text-gray-400">
                  +{data.actions.length - 3} more actions
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* 动作类型和状态 */}
        <div className="mt-2 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {/* 动作类型图标 */}
            {data.metadata?.actionType === 'api' && (
              <div className="flex items-center text-xs text-blue-600">
                <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M12.316 3.051a1 1 0 01.633 1.265l-4 12a1 1 0 11-1.898-.632l4-12a1 1 0 011.265-.633zM5.707 6.293a1 1 0 010 1.414L3.414 10l2.293 2.293a1 1 0 11-1.414 1.414l-3-3a1 1 0 010-1.414l3-3a1 1 0 011.414 0zm8.586 0a1 1 0 011.414 0l3 3a1 1 0 010 1.414l-3 3a1 1 0 11-1.414-1.414L16.586 10l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
                API
              </div>
            )}
            
            {data.metadata?.actionType === 'database' && (
              <div className="flex items-center text-xs text-green-600">
                <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z" />
                </svg>
                DB
              </div>
            )}
            
            {data.metadata?.actionType === 'webhook' && (
              <div className="flex items-center text-xs text-orange-600">
                <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M4 2a2 2 0 00-2 2v11a2 2 0 002 2h12a2 2 0 002-2V4a2 2 0 00-2-2H4zm3 2h2v5L7 7V4zm8 0v10h-2V4h2zM9 9h2v8H9V9z" clipRule="evenodd" />
                </svg>
                Hook
              </div>
            )}
            
            {data.metadata?.actionType === 'script' && (
              <div className="flex items-center text-xs text-indigo-600">
                <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M12.316 3.051a1 1 0 01.633 1.265l-4 12a1 1 0 11-1.898-.632l4-12a1 1 0 011.265-.633z" clipRule="evenodd" />
                </svg>
                Script
              </div>
            )}
          </div>
          
          {/* 执行状态 */}
          <div className="flex items-center space-x-1">
            {data.metadata?.async && (
              <span className="text-xs bg-blue-100 text-blue-700 px-1 py-0.5 rounded">
                Async
              </span>
            )}
            
            {data.metadata?.retry && (
              <span className="text-xs bg-yellow-100 text-yellow-700 px-1 py-0.5 rounded">
                Retry
              </span>
            )}
            
            {data.metadata?.timeout && (
              <span className="text-xs text-gray-500">
                {data.metadata.timeout}s
              </span>
            )}
          </div>
        </div>
        
        {/* 变量和参数 */}
        {data.variables && Object.keys(data.variables).length > 0 && (
          <div className="mt-2">
            <div className="text-xs font-medium text-gray-600 mb-1">Variables:</div>
            <div className="flex flex-wrap gap-1">
              {Object.keys(data.variables).slice(0, 4).map((key) => (
                <span key={key} className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                  ${key}
                </span>
              ))}
              {Object.keys(data.variables).length > 4 && (
                <span className="text-xs text-gray-400">
                  +{Object.keys(data.variables).length - 4}
                </span>
              )}
            </div>
          </div>
        )}
      </div>
      
      {/* 输出连接点 */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 bg-purple-500 border-2 border-white"
      />
    </div>
  );
};

export default ActionNode;
