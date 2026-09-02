import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface PredictionRequest {
  review_text: string;
  model?: string;
  user_id?: string;
  product_id?: string;
}

export interface PredictionResponse {
  prediction: 'FAKE' | 'GENUINE';
  confidence: number;
  fake_probability: number;
  genuine_probability: number;
  model_used: string;
  semantic_confidence?: number;
  graph_confidence?: number;
  adversarial_confidence?: number;
}

export interface MetricsData {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc?: number;
}

export interface ExperimentResult {
  experiment_id: string;
  model: string;
  dataset: string;
  metrics: MetricsData;
  confusion_matrix?: {
    true_positives: number;
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
  };
  timestamp: string;
}

export interface ModelInfo {
  name: string;
  trained: boolean;
  path?: string;
  metrics?: MetricsData;
}

// API functions
export const api = {
  // Health check
  async healthCheck() {
    const response = await apiClient.get('/health');
    return response.data;
  },

  // Predictions
  async predictRoberta(request: PredictionRequest): Promise<PredictionResponse> {
    const response = await apiClient.post('/api/predict/roberta', request);
    return response.data;
  },

  async predictFull(request: PredictionRequest): Promise<PredictionResponse> {
    const response = await apiClient.post('/api/predict/full', request);
    return response.data;
  },

  async getAvailableModels(): Promise<{ models: ModelInfo[] }> {
    const response = await apiClient.get('/api/predict/models');
    return response.data;
  },

  // Experiments
  async listExperiments(): Promise<any[]> {
    const response = await apiClient.get('/api/experiments/');
    return response.data;
  },

  async getExperiment(experimentId: string): Promise<any> {
    const response = await apiClient.get(`/api/experiments/${experimentId}`);
    return response.data;
  },

  // Metrics
  async getModelComparison(): Promise<{ models: any[] }> {
    const response = await apiClient.get('/api/metrics/comparison');
    return response.data;
  },

  async getPaperReference(): Promise<any> {
    const response = await apiClient.get('/api/metrics/paper_reference');
    return response.data;
  },

  // Datasets
  async listDatasets(): Promise<{ datasets: string[], primary: string }> {
    const response = await apiClient.get('/api/datasets/');
    return response.data;
  },
};

export default apiClient;
