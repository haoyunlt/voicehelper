/**
 * 条件节点组件
 * 用于条件判断和分支逻辑的节点
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { DialogNodeData } from '../VisualDialogEditor';

interface ConditionNodeProps extends NodeProps {
  data: DialogNodeData;
}

const ConditionNode: React.FC<ConditionNodeProps> = ({ data, selected }) => {
  return (
    <div className={`
      bg-white border-2 rounded-lg shadow-lg min-w-[180px] max-w-[280px]
      ${selected ? 'border-yellow-500' : 'border-gray-300'}
      hover:border-yellow-400 transition-colors
    `}>
      {/* 输入连接点 */}
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 bg-yellow-500 border-2 border-white"
      />
      
      {/* 节点头部 */}
      <div className="bg-yellow-500 text-white px-3 py-2 rounded-t-lg flex items-center">
        <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
        </svg>
        <span className="text-sm font-medium">Condition</span>
      </div>
      
      {/* 节点内容 */}
      <div className="p-3">
        <div className="mb-2">
          <div className="text-sm font-medium text-gray-700 mb-1">
            {data.label || 'Untitled Condition'}
          </div>
          <div className="text-xs text-gray-600 bg-gray-50 p-2 rounded border">
            {data.content || 'No condition logic defined'}
          </div>
        </div>
        
        {/* 条件列表 */}
        {data.conditions && data.conditions.length > 0 && (
          <div className="mt-2">
            <div className="text-xs font-medium text-gray-600 mb-1">Rules:</div>
            <div className="space-y-1">
              {data.conditions.slice(0, 3).map((condition, index) => (
                <div key={index} className="text-xs bg-yellow-50 px-2 py-1 rounded border-l-2 border-yellow-300">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">
                      {condition.length > 30 ? `${condition.substring(0, 30)}...` : condition}
                    </span>
                    <span className="text-yellow-600 font-mono text-xs">
                      {index === 0 ? 'IF' : 'ELIF'}
                    </span>
                  </div>
                </div>
              ))}
              {data.conditions.length > 3 && (
                <div className="text-xs text-gray-400">
                  +{data.conditions.length - 3} more conditions
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* 变量和操作符 */}
        <div className="mt-2">
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center space-x-1">
              {data.variables && Object.keys(data.variables).length > 0 && (
                <span className="bg-blue-100 text-blue-700 px-2 py-1 rounded">
                  {Object.keys(data.variables).length} vars
                </span>
              )}
              
              {data.metadata?.operator && (
                <span className="bg-purple-100 text-purple-700 px-2 py-1 rounded font-mono">
                  {data.metadata.operator}
                </span>
              )}
            </div>
            
            {/* 条件类型 */}
            <div className="text-gray-500">
              {data.metadata?.conditionType || 'simple'}
            </div>
          </div>
        </div>
      </div>
      
      {/* 多个输出连接点 */}
      <div className="relative">
        {/* True 分支 */}
        <Handle
          type="source"
          position={Position.Bottom}
          id="true"
          className="w-3 h-3 bg-green-500 border-2 border-white"
          style={{ left: '25%' }}
        />
        
        {/* False 分支 */}
        <Handle
          type="source"
          position={Position.Bottom}
          id="false"
          className="w-3 h-3 bg-red-500 border-2 border-white"
          style={{ left: '75%' }}
        />
        
        {/* 分支标签 */}
        <div className="absolute -bottom-6 left-0 right-0 flex justify-between px-4">
          <span className="text-xs text-green-600 font-medium">True</span>
          <span className="text-xs text-red-600 font-medium">False</span>
        </div>
      </div>
    </div>
  );
};

export default ConditionNode;
