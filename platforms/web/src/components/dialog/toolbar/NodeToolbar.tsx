/**
 * 节点工具栏组件
 * 提供添加节点、保存、导出等功能
 */

import React, { useRef } from 'react';
import { DialogNodeData } from '../VisualDialogEditor';

interface NodeToolbarProps {
  onAddNode: (type: DialogNodeData['type'], position?: { x: number; y: number }) => void;
  onSave: () => void;
  onExport: () => void;
  onImport: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onValidate: () => Promise<boolean>;
  isValidating: boolean;
  validationErrors: string[];
  readOnly?: boolean;
}

const NodeToolbar: React.FC<NodeToolbarProps> = ({
  onAddNode,
  onSave,
  onExport,
  onImport,
  onValidate,
  isValidating,
  validationErrors,
  readOnly = false,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const nodeTypes = [
    {
      type: 'start' as const,
      label: 'Start',
      icon: '▶️',
      color: 'bg-green-500',
      description: 'Conversation entry point',
    },
    {
      type: 'intent' as const,
      label: 'Intent',
      icon: '🎯',
      color: 'bg-blue-500',
      description: 'Recognize user intent',
    },
    {
      type: 'response' as const,
      label: 'Response',
      icon: '💬',
      color: 'bg-green-500',
      description: 'Bot response message',
    },
    {
      type: 'condition' as const,
      label: 'Condition',
      icon: '❓',
      color: 'bg-yellow-500',
      description: 'Conditional logic',
    },
    {
      type: 'action' as const,
      label: 'Action',
      icon: '⚡',
      color: 'bg-purple-500',
      description: 'Execute operations',
    },
    {
      type: 'end' as const,
      label: 'End',
      icon: '⏹️',
      color: 'bg-red-500',
      description: 'Conversation endpoint',
    },
  ];

  const handleDragStart = (event: React.DragEvent, nodeType: DialogNodeData['type']) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="w-64 bg-white border-r border-gray-200 flex flex-col h-full">
      {/* 工具栏头部 */}
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-800 mb-2">Dialog Builder</h2>
        <p className="text-sm text-gray-600">
          Drag nodes to canvas to build your conversation flow
        </p>
      </div>

      {/* 节点面板 */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-4">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Node Types</h3>
          <div className="space-y-2">
            {nodeTypes.map((nodeType) => (
              <div
                key={nodeType.type}
                className={`
                  p-3 rounded-lg border-2 border-dashed border-gray-300 cursor-grab
                  hover:border-gray-400 hover:bg-gray-50 transition-colors
                  ${readOnly ? 'opacity-50 cursor-not-allowed' : ''}
                `}
                draggable={!readOnly}
                onDragStart={(e) => handleDragStart(e, nodeType.type)}
                onClick={() => !readOnly && onAddNode(nodeType.type)}
              >
                <div className="flex items-center space-x-3">
                  <div className={`
                    w-8 h-8 rounded-full ${nodeType.color} flex items-center justify-center text-white text-sm
                  `}>
                    {nodeType.icon}
                  </div>
                  <div className="flex-1">
                    <div className="text-sm font-medium text-gray-800">
                      {nodeType.label}
                    </div>
                    <div className="text-xs text-gray-500">
                      {nodeType.description}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 快捷操作 */}
        <div className="p-4 border-t border-gray-200">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Quick Actions</h3>
          <div className="space-y-2">
            <button
              onClick={() => onAddNode('start')}
              disabled={readOnly}
              className="w-full text-left p-2 text-sm text-gray-600 hover:bg-gray-50 rounded disabled:opacity-50"
            >
              ➕ Add Start Node
            </button>
            <button
              onClick={() => onAddNode('response')}
              disabled={readOnly}
              className="w-full text-left p-2 text-sm text-gray-600 hover:bg-gray-50 rounded disabled:opacity-50"
            >
              💬 Add Response
            </button>
            <button
              onClick={() => onAddNode('condition')}
              disabled={readOnly}
              className="w-full text-left p-2 text-sm text-gray-600 hover:bg-gray-50 rounded disabled:opacity-50"
            >
              ❓ Add Condition
            </button>
          </div>
        </div>
      </div>

      {/* 底部操作栏 */}
      <div className="p-4 border-t border-gray-200 space-y-2">
        {/* 验证按钮 */}
        <button
          onClick={onValidate}
          disabled={isValidating}
          className={`
            w-full px-4 py-2 text-sm font-medium rounded-lg transition-colors
            ${validationErrors.length > 0
              ? 'bg-red-100 text-red-700 hover:bg-red-200'
              : 'bg-green-100 text-green-700 hover:bg-green-200'
            }
            disabled:opacity-50 disabled:cursor-not-allowed
          `}
        >
          {isValidating ? (
            <div className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Validating...
            </div>
          ) : (
            <>
              ✓ Validate Flow
              {validationErrors.length > 0 && (
                <span className="ml-2 bg-red-200 text-red-800 text-xs px-2 py-1 rounded-full">
                  {validationErrors.length}
                </span>
              )}
            </>
          )}
        </button>

        {/* 保存按钮 */}
        {!readOnly && (
          <button
            onClick={onSave}
            className="w-full bg-blue-600 text-white px-4 py-2 text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors"
          >
            💾 Save Flow
          </button>
        )}

        {/* 导入/导出 */}
        <div className="flex space-x-2">
          <button
            onClick={handleImportClick}
            disabled={readOnly}
            className="flex-1 bg-gray-100 text-gray-700 px-3 py-2 text-sm font-medium rounded-lg hover:bg-gray-200 transition-colors disabled:opacity-50"
          >
            📁 Import
          </button>
          <button
            onClick={onExport}
            className="flex-1 bg-gray-100 text-gray-700 px-3 py-2 text-sm font-medium rounded-lg hover:bg-gray-200 transition-colors"
          >
            📤 Export
          </button>
        </div>

        {/* 隐藏的文件输入 */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          onChange={onImport}
          className="hidden"
        />

        {/* 统计信息 */}
        <div className="text-xs text-gray-500 pt-2 border-t border-gray-100">
          <div className="flex justify-between">
            <span>Nodes</span>
            <span>Edges</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NodeToolbar;
