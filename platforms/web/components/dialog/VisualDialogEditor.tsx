/**
 * 可视化对话流编辑器
 * 基于React Flow实现拖拽式对话设计
 * 参考Botpress和Microsoft Bot Framework的设计
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  BackgroundVariant,
  MiniMap,
  Panel,
  NodeTypes,
  EdgeTypes,
  ReactFlowProvider,
  useReactFlow,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';

// 自定义节点类型
import IntentNode from './nodes/IntentNode';
import ResponseNode from './nodes/ResponseNode';
import ConditionNode from './nodes/ConditionNode';
import ActionNode from './nodes/ActionNode';
import StartNode from './nodes/StartNode';
import EndNode from './nodes/EndNode';

// 工具栏和属性面板
import NodeToolbar from './toolbar/NodeToolbar';
import PropertyPanel from './panels/PropertyPanel';
import FlowValidation from './validation/FlowValidation';

// 类型定义
export interface DialogNodeData {
  type: 'start' | 'intent' | 'response' | 'condition' | 'action' | 'end';
  label: string;
  content: string;
  conditions?: string[] | undefined;
  responses?: string[] | undefined;
  actions?: string[] | undefined;
  variables?: Record<string, any> | undefined;
  metadata?: Record<string, any> | undefined;
}

export interface DialogNode extends Node {
  data: DialogNodeData;
}

export interface DialogEdge {
  id: string;
  source: string | null;
  target: string | null;
  sourceHandle?: string | null;
  targetHandle?: string | null;
  type?: string;
  animated?: boolean;
  data?: {
    condition?: string;
    probability?: number;
    label?: string;
  };
}

// 自定义节点类型映射
const nodeTypes: NodeTypes = {
  start: StartNode,
  intent: IntentNode,
  response: ResponseNode,
  condition: ConditionNode,
  action: ActionNode,
  end: EndNode,
};

// 自定义边类型
const edgeTypes: EdgeTypes = {
  // 可以添加自定义边类型
};

// 默认边样式
const defaultEdgeOptions = {
  animated: true,
  markerEnd: {
    type: MarkerType.ArrowClosed,
    width: 20,
    height: 20,
    color: '#FF0072',
  },
  style: {
    strokeWidth: 2,
    stroke: '#FF0072',
  },
};

interface VisualDialogEditorProps {
  initialFlow?: {
    nodes: DialogNode[];
    edges: DialogEdge[];
  };
  onFlowChange?: (nodes: DialogNode[], edges: DialogEdge[]) => void;
  onSave?: (flow: any) => void;
  readOnly?: boolean;
}

const VisualDialogEditor: React.FC<VisualDialogEditorProps> = ({
  initialFlow,
  onFlowChange,
  onSave,
  readOnly = false,
}) => {
  const [nodes, setNodes, onNodesChange] = useNodesState(
    (initialFlow?.nodes || []) as Node<DialogNodeData>[]
  );
  const [edges, setEdges, onEdgesChange] = useEdgesState(
    (initialFlow?.edges || []) as Edge[]
  );
  
  const [selectedNode, setSelectedNode] = useState<DialogNode | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<DialogEdge | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [showPropertyPanel, setShowPropertyPanel] = useState(true);
  
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { project, getViewport } = useReactFlow();

  // 监听流程变化
  useEffect(() => {
    onFlowChange?.(nodes as DialogNode[], edges as DialogEdge[]);
  }, [nodes, edges, onFlowChange]);

  // 连接节点
  const onConnect = useCallback(
    (params: Connection) => {
      const newEdge: DialogEdge = {
        ...params,
        id: `edge-${Date.now()}`,
        type: 'default',
        animated: true,
        data: {
          label: '',
          condition: '',
          probability: 1.0,
        },
      };
      setEdges((eds) => addEdge(newEdge as Edge, eds));
    },
    [setEdges]
  );

  // 添加节点
  const addNode = useCallback(
    (type: DialogNodeData['type'], position?: { x: number; y: number }) => {
      if (readOnly) return;

      const viewport = getViewport();
      const defaultPosition = position || {
        x: Math.random() * 400 + viewport.x,
        y: Math.random() * 400 + viewport.y,
      };

      const newNode: DialogNode = {
        id: `${type}-${Date.now()}`,
        type,
        position: defaultPosition,
        data: {
          type,
          label: `New ${type.charAt(0).toUpperCase() + type.slice(1)}`,
          content: '',
          conditions: type === 'condition' ? [''] : undefined,
          responses: type === 'response' ? [''] : undefined,
          actions: type === 'action' ? [''] : undefined,
          variables: {},
          metadata: {},
        },
      };

      setNodes((nds) => [...nds, newNode]);
      setSelectedNode(newNode);
    },
    [readOnly, getViewport, setNodes]
  );

  // 删除节点
  const deleteNode = useCallback(
    (nodeId: string) => {
      if (readOnly) return;
      
      setNodes((nds) => nds.filter((node) => node.id !== nodeId));
      setEdges((eds) => eds.filter((edge) => 
        edge.source !== nodeId && edge.target !== nodeId
      ));
      
      if (selectedNode?.id === nodeId) {
        setSelectedNode(null);
      }
    },
    [readOnly, selectedNode, setNodes, setEdges]
  );

  // 更新节点数据
  const updateNodeData = useCallback(
    (nodeId: string, newData: Partial<DialogNodeData>) => {
      if (readOnly) return;
      
      setNodes((nds) =>
        nds.map((node) =>
          node.id === nodeId
            ? { ...node, data: { ...node.data, ...newData } }
            : node
        )
      );
    },
    [readOnly, setNodes]
  );

  // 更新边数据
  const updateEdgeData = useCallback(
    (edgeId: string, newData: any) => {
      if (readOnly) return;
      
      setEdges((eds) =>
        eds.map((edge) =>
          edge.id === edgeId
            ? { ...edge, data: { ...edge.data, ...newData } }
            : edge
        )
      );
    },
    [readOnly, setEdges]
  );

  // 验证流程
  const validateFlow = useCallback(async () => {
    setIsValidating(true);
    
    try {
      const validator = new FlowValidation();
      const errors = await validator.validate(nodes, edges);
      setValidationErrors(errors);
      
      return errors.length === 0;
    } catch (error) {
      console.error('Flow validation error:', error);
      setValidationErrors(['Validation failed']);
      return false;
    } finally {
      setIsValidating(false);
    }
  }, [nodes, edges]);

  // 保存流程
  const handleSave = useCallback(async () => {
    if (readOnly) return;
    
    const isValid = await validateFlow();
    if (!isValid) {
      alert('Please fix validation errors before saving');
      return;
    }

    const flowData = {
      nodes: nodes.map(node => ({
        id: node.id,
        type: node.type,
        position: node.position,
        data: node.data,
      })),
      edges: edges.map(edge => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        data: edge.data,
      })),
      metadata: {
        version: '1.0',
        createdAt: new Date().toISOString(),
        nodeCount: nodes.length,
        edgeCount: edges.length,
      },
    };

    onSave?.(flowData);
  }, [readOnly, nodes, edges, validateFlow, onSave]);

  // 导出流程
  const exportFlow = useCallback(() => {
    const flowData = {
      nodes,
      edges,
      viewport: getViewport(),
    };
    
    const dataStr = JSON.stringify(flowData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `dialog-flow-${Date.now()}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  }, [nodes, edges, getViewport]);

  // 导入流程
  const importFlow = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    if (readOnly) return;
    
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const flowData = JSON.parse(e.target?.result as string);
        setNodes(flowData.nodes || []);
        setEdges(flowData.edges || []);
      } catch (error) {
        console.error('Import error:', error);
        alert('Failed to import flow file');
      }
    };
    reader.readAsText(file);
  }, [readOnly, setNodes, setEdges]);

  // 节点点击处理
  const onNodeClick = useCallback(
    (event: React.MouseEvent, node: DialogNode) => {
      setSelectedNode(node);
      setSelectedEdge(null);
    },
    []
  );

  // 边点击处理
  const onEdgeClick = useCallback(
    (event: React.MouseEvent, edge: DialogEdge) => {
      setSelectedEdge(edge);
      setSelectedNode(null);
    },
    []
  );

  // 拖拽添加节点
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const reactFlowBounds = reactFlowWrapper.current?.getBoundingClientRect();
      if (!reactFlowBounds) return;

      const type = event.dataTransfer.getData('application/reactflow') as DialogNodeData['type'];
      if (!type) return;

      const position = project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      addNode(type, position);
    },
    [project, addNode]
  );

  return (
    <div className="h-screen flex bg-gray-50">
      {/* 工具栏 */}
      <NodeToolbar
        onAddNode={addNode}
        onSave={handleSave}
        onExport={exportFlow}
        onImport={importFlow}
        onValidate={validateFlow}
        isValidating={isValidating}
        validationErrors={validationErrors}
        readOnly={readOnly}
      />

      {/* 主编辑区域 */}
      <div className="flex-1 relative" ref={reactFlowWrapper}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          onEdgeClick={onEdgeClick}
          onDrop={onDrop}
          onDragOver={onDragOver}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          defaultEdgeOptions={defaultEdgeOptions}
          fitView
          attributionPosition="bottom-left"
        >
          <Controls />
          <MiniMap />
          <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
          
          {/* 顶部面板 */}
          <Panel position="top-center">
            <div className="bg-white rounded-lg shadow-lg px-4 py-2 flex items-center space-x-4">
              <span className="text-sm font-medium">
                Nodes: {nodes.length} | Edges: {edges.length}
              </span>
              
              {validationErrors.length > 0 && (
                <div className="flex items-center text-red-600">
                  <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  <span className="text-sm">{validationErrors.length} errors</span>
                </div>
              )}
              
              <button
                onClick={() => setShowPropertyPanel(!showPropertyPanel)}
                className="text-sm bg-blue-500 text-white px-3 py-1 rounded hover:bg-blue-600"
              >
                {showPropertyPanel ? 'Hide' : 'Show'} Properties
              </button>
            </div>
          </Panel>
        </ReactFlow>
      </div>

      {/* 属性面板 */}
      {showPropertyPanel && (
        <PropertyPanel
          selectedNode={selectedNode}
          selectedEdge={selectedEdge}
          onUpdateNode={updateNodeData}
          onUpdateEdge={updateEdgeData}
          onDeleteNode={deleteNode}
          readOnly={readOnly}
        />
      )}
    </div>
  );
};

// 包装组件以提供ReactFlow上下文
const VisualDialogEditorWrapper: React.FC<VisualDialogEditorProps> = (props) => {
  return (
    <ReactFlowProvider>
      <VisualDialogEditor {...props} />
    </ReactFlowProvider>
  );
};

export default VisualDialogEditorWrapper;
