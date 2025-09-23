/**
 * 对话流验证器
 * 验证对话流的完整性和正确性
 */

import { DialogNode, DialogEdge } from '../VisualDialogEditor';

export interface ValidationError {
  type: 'error' | 'warning' | 'info';
  message: string;
  nodeId?: string;
  edgeId?: string;
  severity: 'high' | 'medium' | 'low';
}

export class FlowValidation {
  async validate(nodes: DialogNode[], edges: DialogEdge[]): Promise<string[]> {
    const errors: ValidationError[] = [];

    // 基础结构验证
    errors.push(...this.validateBasicStructure(nodes, edges));
    
    // 节点连接验证
    errors.push(...this.validateConnections(nodes, edges));
    
    // 节点内容验证
    errors.push(...this.validateNodeContent(nodes));
    
    // 流程逻辑验证
    errors.push(...this.validateFlowLogic(nodes, edges));
    
    // 可达性验证
    errors.push(...this.validateReachability(nodes, edges));

    // 返回错误消息
    return errors
      .filter(error => error.type === 'error')
      .map(error => error.message);
  }

  private validateBasicStructure(nodes: DialogNode[], edges: DialogEdge[]): ValidationError[] {
    const errors: ValidationError[] = [];

    // 检查是否有节点
    if (nodes.length === 0) {
      errors.push({
        type: 'error',
        message: 'Flow must contain at least one node',
        severity: 'high'
      });
      return errors;
    }

    // 检查开始节点
    const startNodes = nodes.filter(node => node.data.type === 'start');
    if (startNodes.length === 0) {
      errors.push({
        type: 'error',
        message: 'Flow must have exactly one start node',
        severity: 'high'
      });
    } else if (startNodes.length > 1) {
      errors.push({
        type: 'error',
        message: 'Flow cannot have multiple start nodes',
        severity: 'high'
      });
    }

    // 检查结束节点
    const endNodes = nodes.filter(node => node.data.type === 'end');
    if (endNodes.length === 0) {
      errors.push({
        type: 'warning',
        message: 'Flow should have at least one end node',
        severity: 'medium'
      });
    }

    // 检查节点ID唯一性
    const nodeIds = new Set();
    nodes.forEach(node => {
      if (nodeIds.has(node.id)) {
        errors.push({
          type: 'error',
          message: `Duplicate node ID: ${node.id}`,
          nodeId: node.id,
          severity: 'high'
        });
      }
      nodeIds.add(node.id);
    });

    // 检查边ID唯一性
    const edgeIds = new Set();
    edges.forEach(edge => {
      if (edgeIds.has(edge.id)) {
        errors.push({
          type: 'error',
          message: `Duplicate edge ID: ${edge.id}`,
          edgeId: edge.id,
          severity: 'high'
        });
      }
      edgeIds.add(edge.id);
    });

    return errors;
  }

  private validateConnections(nodes: DialogNode[], edges: DialogEdge[]): ValidationError[] {
    const errors: ValidationError[] = [];
    const nodeIds = new Set(nodes.map(node => node.id));

    // 检查边的源节点和目标节点是否存在
    edges.forEach(edge => {
      if (edge.source && !nodeIds.has(edge.source)) {
        errors.push({
          type: 'error',
          message: `Edge ${edge.id} references non-existent source node: ${edge.source}`,
          edgeId: edge.id,
          severity: 'high'
        });
      }
      
      if (edge.target && !nodeIds.has(edge.target)) {
        errors.push({
          type: 'error',
          message: `Edge ${edge.id} references non-existent target node: ${edge.target}`,
          edgeId: edge.id,
          severity: 'high'
        });
      }
    });

    // 检查孤立节点
    const connectedNodes = new Set();
    edges.forEach(edge => {
      connectedNodes.add(edge.source);
      connectedNodes.add(edge.target);
    });

    nodes.forEach(node => {
      if (!connectedNodes.has(node.id) && node.data.type !== 'start' && node.data.type !== 'end') {
        errors.push({
          type: 'warning',
          message: `Node "${node.data.label || node.id}" is not connected to any other nodes`,
          nodeId: node.id,
          severity: 'medium'
        });
      }
    });

    // 检查开始节点的输入连接
    const startNodes = nodes.filter(node => node.data.type === 'start');
    startNodes.forEach(startNode => {
      const incomingEdges = edges.filter(edge => edge.target === startNode.id);
      if (incomingEdges.length > 0) {
        errors.push({
          type: 'warning',
          message: 'Start node should not have incoming connections',
          nodeId: startNode.id,
          severity: 'medium'
        });
      }
    });

    // 检查结束节点的输出连接
    const endNodes = nodes.filter(node => node.data.type === 'end');
    endNodes.forEach(endNode => {
      const outgoingEdges = edges.filter(edge => edge.source === endNode.id);
      if (outgoingEdges.length > 0) {
        errors.push({
          type: 'warning',
          message: 'End node should not have outgoing connections',
          nodeId: endNode.id,
          severity: 'medium'
        });
      }
    });

    return errors;
  }

  private validateNodeContent(nodes: DialogNode[]): ValidationError[] {
    const errors: ValidationError[] = [];

    nodes.forEach(node => {
      // 检查节点标签
      if (!node.data.label || node.data.label.trim() === '') {
        errors.push({
          type: 'warning',
          message: `Node ${node.id} has no label`,
          nodeId: node.id,
          severity: 'low'
        });
      }

      // 根据节点类型验证内容
      switch (node.data.type) {
        case 'intent':
          if (!node.data.content || node.data.content.trim() === '') {
            errors.push({
              type: 'error',
              message: `Intent node "${node.data.label}" must have intent pattern defined`,
              nodeId: node.id,
              severity: 'high'
            });
          }
          break;

        case 'response':
          if (!node.data.content && (!node.data.responses || node.data.responses.length === 0)) {
            errors.push({
              type: 'error',
              message: `Response node "${node.data.label}" must have response content`,
              nodeId: node.id,
              severity: 'high'
            });
          }
          
          // 检查空的响应变体
          if (node.data.responses) {
            const emptyResponses = node.data.responses.filter(response => !response || response.trim() === '');
            if (emptyResponses.length > 0) {
              errors.push({
                type: 'warning',
                message: `Response node "${node.data.label}" has empty response variants`,
                nodeId: node.id,
                severity: 'medium'
              });
            }
          }
          break;

        case 'condition':
          if (!node.data.conditions || node.data.conditions.length === 0) {
            errors.push({
              type: 'error',
              message: `Condition node "${node.data.label}" must have at least one condition`,
              nodeId: node.id,
              severity: 'high'
            });
          }
          
          // 检查空的条件
          if (node.data.conditions) {
            const emptyConditions = node.data.conditions.filter(condition => !condition || condition.trim() === '');
            if (emptyConditions.length > 0) {
              errors.push({
                type: 'warning',
                message: `Condition node "${node.data.label}" has empty conditions`,
                nodeId: node.id,
                severity: 'medium'
              });
            }
          }
          break;

        case 'action':
          if (!node.data.actions || node.data.actions.length === 0) {
            errors.push({
              type: 'error',
              message: `Action node "${node.data.label}" must have at least one action`,
              nodeId: node.id,
              severity: 'high'
            });
          }
          
          // 检查空的动作
          if (node.data.actions) {
            const emptyActions = node.data.actions.filter(action => !action || action.trim() === '');
            if (emptyActions.length > 0) {
              errors.push({
                type: 'warning',
                message: `Action node "${node.data.label}" has empty actions`,
                nodeId: node.id,
                severity: 'medium'
              });
            }
          }
          break;
      }
    });

    return errors;
  }

  private validateFlowLogic(nodes: DialogNode[], edges: DialogEdge[]): ValidationError[] {
    const errors: ValidationError[] = [];

    // 检查条件节点的分支
    const conditionNodes = nodes.filter(node => node.data.type === 'condition');
    conditionNodes.forEach(conditionNode => {
      const outgoingEdges = edges.filter(edge => edge.source === conditionNode.id);
      
      if (outgoingEdges.length === 0) {
        errors.push({
          type: 'error',
          message: `Condition node "${conditionNode.data.label}" has no outgoing connections`,
          nodeId: conditionNode.id,
          severity: 'high'
        });
      } else if (outgoingEdges.length === 1) {
        errors.push({
          type: 'warning',
          message: `Condition node "${conditionNode.data.label}" should have at least 2 branches (true/false)`,
          nodeId: conditionNode.id,
          severity: 'medium'
        });
      }
      
      // 检查是否有true和false分支
      const hasTrueBranch = outgoingEdges.some(edge => edge.sourceHandle === 'true');
      const hasFalseBranch = outgoingEdges.some(edge => edge.sourceHandle === 'false');
      
      if (!hasTrueBranch) {
        errors.push({
          type: 'warning',
          message: `Condition node "${conditionNode.data.label}" is missing true branch`,
          nodeId: conditionNode.id,
          severity: 'medium'
        });
      }
      
      if (!hasFalseBranch) {
        errors.push({
          type: 'warning',
          message: `Condition node "${conditionNode.data.label}" is missing false branch`,
          nodeId: conditionNode.id,
          severity: 'medium'
        });
      }
    });

    // 检查循环引用
    const cycles = this.detectCycles(nodes, edges);
    cycles.forEach(cycle => {
      errors.push({
        type: 'warning',
        message: `Potential infinite loop detected: ${cycle.join(' -> ')}`,
        severity: 'medium'
      });
    });

    return errors;
  }

  private validateReachability(nodes: DialogNode[], edges: DialogEdge[]): ValidationError[] {
    const errors: ValidationError[] = [];
    
    const startNodes = nodes.filter(node => node.data.type === 'start');
    if (startNodes.length === 0) return errors;

    const reachableNodes = this.getReachableNodes(startNodes[0]?.id || '', edges);
    
    nodes.forEach(node => {
      if (!reachableNodes.has(node.id) && node.data.type !== 'start') {
        errors.push({
          type: 'warning',
          message: `Node "${node.data.label}" is not reachable from start node`,
          nodeId: node.id,
          severity: 'medium'
        });
      }
    });

    return errors;
  }

  private getReachableNodes(startNodeId: string, edges: DialogEdge[]): Set<string> {
    const reachable = new Set<string>();
    const queue = [startNodeId];
    
    while (queue.length > 0) {
      const currentId = queue.shift()!;
      if (reachable.has(currentId)) continue;
      
      reachable.add(currentId);
      
      const outgoingEdges = edges.filter(edge => edge.source === currentId);
      outgoingEdges.forEach(edge => {
        if (edge.target && !reachable.has(edge.target)) {
          queue.push(edge.target);
        }
      });
    }
    
    return reachable;
  }

  private detectCycles(nodes: DialogNode[], edges: DialogEdge[]): string[][] {
    const cycles: string[][] = [];
    const visited = new Set<string>();
    const recursionStack = new Set<string>();
    
    const dfs = (nodeId: string, path: string[]): boolean => {
      if (recursionStack.has(nodeId)) {
        // 找到循环
        const cycleStart = path.indexOf(nodeId);
        if (cycleStart !== -1) {
          cycles.push([...path.slice(cycleStart), nodeId]);
        }
        return true;
      }
      
      if (visited.has(nodeId)) return false;
      
      visited.add(nodeId);
      recursionStack.add(nodeId);
      path.push(nodeId);
      
      const outgoingEdges = edges.filter(edge => edge.source === nodeId);
      for (const edge of outgoingEdges) {
        if (edge.target && dfs(edge.target, path)) {
          // 如果找到循环，继续搜索其他路径
        }
      }
      
      recursionStack.delete(nodeId);
      path.pop();
      
      return false;
    };
    
    nodes.forEach(node => {
      if (!visited.has(node.id)) {
        dfs(node.id, []);
      }
    });
    
    return cycles;
  }
}

export default FlowValidation;
