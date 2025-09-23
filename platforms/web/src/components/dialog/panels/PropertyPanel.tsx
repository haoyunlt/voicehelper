/**
 * 属性面板组件
 * 用于编辑选中节点或边的属性
 */

import React, { useState, useEffect } from 'react';
import { DialogNode, DialogEdge, DialogNodeData } from '../VisualDialogEditor';

interface PropertyPanelProps {
  selectedNode: DialogNode | null;
  selectedEdge: DialogEdge | null;
  onUpdateNode: (nodeId: string, data: Partial<DialogNodeData>) => void;
  onUpdateEdge: (edgeId: string, data: any) => void;
  onDeleteNode: (nodeId: string) => void;
  readOnly?: boolean;
}

const PropertyPanel: React.FC<PropertyPanelProps> = ({
  selectedNode,
  selectedEdge,
  onUpdateNode,
  onUpdateEdge,
  onDeleteNode,
  readOnly = false,
}) => {
  const [localData, setLocalData] = useState<any>({});

  // 同步选中项数据
  useEffect(() => {
    if (selectedNode) {
      setLocalData(selectedNode.data);
    } else if (selectedEdge) {
      setLocalData(selectedEdge.data || {});
    } else {
      setLocalData({});
    }
  }, [selectedNode, selectedEdge]);

  // 更新本地数据
  const updateLocalData = (key: string, value: any) => {
    const newData = { ...localData, [key]: value };
    setLocalData(newData);

    // 立即同步到父组件
    if (selectedNode) {
      onUpdateNode(selectedNode.id, newData);
    } else if (selectedEdge) {
      onUpdateEdge(selectedEdge.id, newData);
    }
  };

  // 更新数组类型的数据
  const updateArrayData = (key: string, index: number, value: string) => {
    const array = localData[key] || [];
    const newArray = [...array];
    newArray[index] = value;
    updateLocalData(key, newArray);
  };

  // 添加数组项
  const addArrayItem = (key: string) => {
    const array = localData[key] || [];
    updateLocalData(key, [...array, '']);
  };

  // 删除数组项
  const removeArrayItem = (key: string, index: number) => {
    const array = localData[key] || [];
    const newArray = array.filter((_item: any, i: number) => i !== index);
    updateLocalData(key, newArray);
  };

  // 更新变量
  const updateVariable = (key: string, value: any) => {
    const variables = localData.variables || {};
    updateLocalData('variables', { ...variables, [key]: value });
  };

  // 删除变量
  const removeVariable = (key: string) => {
    const variables = localData.variables || {};
    const newVariables = { ...variables };
    delete newVariables[key];
    updateLocalData('variables', newVariables);
  };

  if (!selectedNode && !selectedEdge) {
    return (
      <div className="w-80 bg-white border-l border-gray-200 p-4">
        <div className="text-center text-gray-500 mt-8">
          <svg className="w-12 h-12 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.122 2.122" />
          </svg>
          <p className="text-sm">Select a node or edge to edit its properties</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-80 bg-white border-l border-gray-200 flex flex-col h-full">
      {/* 头部 */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-800">
            {selectedNode ? 'Node Properties' : 'Edge Properties'}
          </h3>
          {selectedNode && !readOnly && (
            <button
              onClick={() => onDeleteNode(selectedNode.id)}
              className="text-red-600 hover:text-red-800 p-1"
              title="Delete Node"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          )}
        </div>
        {selectedNode && (
          <p className="text-sm text-gray-600 mt-1">
            Type: {selectedNode.data.type} • ID: {selectedNode.id}
          </p>
        )}
      </div>

      {/* 属性编辑区域 */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {selectedNode && (
          <>
            {/* 基本属性 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Label
              </label>
              <input
                type="text"
                value={localData.label || ''}
                onChange={(e) => updateLocalData('label', e.target.value)}
                disabled={readOnly}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
                placeholder="Enter node label"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Content
              </label>
              <textarea
                value={localData.content || ''}
                onChange={(e) => updateLocalData('content', e.target.value)}
                disabled={readOnly}
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
                placeholder="Enter node content"
              />
            </div>

            {/* 条件节点特有属性 */}
            {selectedNode.data.type === 'condition' && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-700">
                    Conditions
                  </label>
                  {!readOnly && (
                    <button
                      onClick={() => addArrayItem('conditions')}
                      className="text-blue-600 hover:text-blue-800 text-sm"
                    >
                      + Add
                    </button>
                  )}
                </div>
                <div className="space-y-2">
                  {(localData.conditions || []).map((condition: string, index: number) => (
                    <div key={index} className="flex items-center space-x-2">
                      <input
                        type="text"
                        value={condition}
                        onChange={(e) => updateArrayData('conditions', index, e.target.value)}
                        disabled={readOnly}
                        className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
                        placeholder={`Condition ${index + 1}`}
                      />
                      {!readOnly && (
                        <button
                          onClick={() => removeArrayItem('conditions', index)}
                          className="text-red-600 hover:text-red-800"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 响应节点特有属性 */}
            {selectedNode.data.type === 'response' && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-700">
                    Response Variants
                  </label>
                  {!readOnly && (
                    <button
                      onClick={() => addArrayItem('responses')}
                      className="text-blue-600 hover:text-blue-800 text-sm"
                    >
                      + Add
                    </button>
                  )}
                </div>
                <div className="space-y-2">
                  {(localData.responses || []).map((response: string, index: number) => (
                    <div key={index} className="flex items-center space-x-2">
                      <textarea
                        value={response}
                        onChange={(e) => updateArrayData('responses', index, e.target.value)}
                        disabled={readOnly}
                        rows={2}
                        className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
                        placeholder={`Response variant ${index + 1}`}
                      />
                      {!readOnly && (
                        <button
                          onClick={() => removeArrayItem('responses', index)}
                          className="text-red-600 hover:text-red-800"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 动作节点特有属性 */}
            {selectedNode.data.type === 'action' && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-700">
                    Actions
                  </label>
                  {!readOnly && (
                    <button
                      onClick={() => addArrayItem('actions')}
                      className="text-blue-600 hover:text-blue-800 text-sm"
                    >
                      + Add
                    </button>
                  )}
                </div>
                <div className="space-y-2">
                  {(localData.actions || []).map((action: string, index: number) => (
                    <div key={index} className="flex items-center space-x-2">
                      <input
                        type="text"
                        value={action}
                        onChange={(e) => updateArrayData('actions', index, e.target.value)}
                        disabled={readOnly}
                        className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
                        placeholder={`Action ${index + 1}`}
                      />
                      {!readOnly && (
                        <button
                          onClick={() => removeArrayItem('actions', index)}
                          className="text-red-600 hover:text-red-800"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 变量管理 */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm font-medium text-gray-700">
                  Variables
                </label>
                {!readOnly && (
                  <button
                    onClick={() => {
                      const key = prompt('Variable name:');
                      if (key) updateVariable(key, '');
                    }}
                    className="text-blue-600 hover:text-blue-800 text-sm"
                  >
                    + Add
                  </button>
                )}
              </div>
              <div className="space-y-2">
                {Object.entries(localData.variables || {}).map(([key, value]) => (
                  <div key={key} className="flex items-center space-x-2">
                    <input
                      type="text"
                      value={key}
                      disabled
                      className="w-24 px-2 py-1 text-sm border border-gray-300 rounded bg-gray-100"
                    />
                    <input
                      type="text"
                      value={value as string}
                      onChange={(e) => updateVariable(key, e.target.value)}
                      disabled={readOnly}
                      className="flex-1 px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:bg-gray-100"
                      placeholder="Value"
                    />
                    {!readOnly && (
                      <button
                        onClick={() => removeVariable(key)}
                        className="text-red-600 hover:text-red-800"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* 元数据 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Metadata (JSON)
              </label>
              <textarea
                value={JSON.stringify(localData.metadata || {}, null, 2)}
                onChange={(e) => {
                  try {
                    const metadata = JSON.parse(e.target.value);
                    updateLocalData('metadata', metadata);
                  } catch (error) {
                    // 忽略JSON解析错误，用户可能正在编辑
                  }
                }}
                disabled={readOnly}
                rows={4}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono text-sm disabled:bg-gray-100"
                placeholder="{}"
              />
            </div>
          </>
        )}

        {selectedEdge && (
          <>
            {/* 边属性 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Label
              </label>
              <input
                type="text"
                value={localData.label || ''}
                onChange={(e) => updateLocalData('label', e.target.value)}
                disabled={readOnly}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
                placeholder="Edge label"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Condition
              </label>
              <textarea
                value={localData.condition || ''}
                onChange={(e) => updateLocalData('condition', e.target.value)}
                disabled={readOnly}
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
                placeholder="Condition for this transition"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Probability
              </label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.1"
                value={localData.probability || 1}
                onChange={(e) => updateLocalData('probability', parseFloat(e.target.value))}
                disabled={readOnly}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default PropertyPanel;
