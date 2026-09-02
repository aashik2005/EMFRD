import { useEffect, useState } from 'react';
import { Card, Table, Typography, Spin, Alert, Row, Col, Statistic, Tag } from 'antd';
import { api, MetricsData } from '../api/client';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

const { Title, Paragraph, Text } = Typography;

interface ModelResult {
  model: string;
  metrics: MetricsData;
  timestamp?: string;
}

function ExperimentsPage() {
  const [loading, setLoading] = useState(true);
  const [experiments, setExperiments] = useState<any[]>([]);
  const [modelResults, setModelResults] = useState<ModelResult[]>([]);
  const [paperReference, setPaperReference] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);

      // Load experiments list
      const experimentsData = await api.listExperiments();
      setExperiments(experimentsData);

      // Load model comparison
      const comparisonData = await api.getModelComparison();
      setModelResults(comparisonData.models);

      // Load paper reference
      const paperData = await api.getPaperReference();
      setPaperReference(paperData);

      setError(null);
    } catch (err: any) {
      setError(err.message || 'Failed to load experiments');
    } finally {
      setLoading(false);
    }
  };

  const columns = [
    {
      title: 'Model',
      dataIndex: 'model',
      key: 'model',
      render: (model: string) => <Text strong>{model.replace(/_/g, ' ').toUpperCase()}</Text>,
    },
    {
      title: 'Accuracy',
      key: 'accuracy',
      render: (_: any, record: ModelResult) => (
        <span>{(record.metrics.accuracy * 100).toFixed(2)}%</span>
      ),
      sorter: (a: ModelResult, b: ModelResult) => a.metrics.accuracy - b.metrics.accuracy,
    },
    {
      title: 'Precision',
      key: 'precision',
      render: (_: any, record: ModelResult) => (
        <span>{(record.metrics.precision * 100).toFixed(2)}%</span>
      ),
    },
    {
      title: 'Recall',
      key: 'recall',
      render: (_: any, record: ModelResult) => (
        <span>{(record.metrics.recall * 100).toFixed(2)}%</span>
      ),
    },
    {
      title: 'F1 Score',
      key: 'f1',
      render: (_: any, record: ModelResult) => (
        <span>{(record.metrics.f1 * 100).toFixed(2)}%</span>
      ),
      sorter: (a: ModelResult, b: ModelResult) => a.metrics.f1 - b.metrics.f1,
    },
    {
      title: 'ROC-AUC',
      key: 'roc_auc',
      render: (_: any, record: ModelResult) =>
        record.metrics.roc_auc ? (
          <span>{(record.metrics.roc_auc * 100).toFixed(2)}%</span>
        ) : (
          <Text type="secondary">-</Text>
        ),
    },
  ];

  // Prepare chart data
  const chartData = modelResults.map((result) => ({
    name: result.model.replace(/_/g, ' '),
    Accuracy: (result.metrics.accuracy * 100).toFixed(1),
    Precision: (result.metrics.precision * 100).toFixed(1),
    Recall: (result.metrics.recall * 100).toFixed(1),
    'F1 Score': (result.metrics.f1 * 100).toFixed(1),
  }));

  // Add paper reference if available
  if (paperReference && paperReference.roberta_baseline) {
    const paperBaseline = {
      name: 'Paper Reference (RoBERTa)',
      Accuracy: (paperReference.roberta_baseline.accuracy * 100).toFixed(1),
      Precision: (paperReference.roberta_baseline.precision * 100).toFixed(1),
      Recall: (paperReference.roberta_baseline.recall * 100).toFixed(1),
      'F1 Score': (paperReference.roberta_baseline.f1 * 100).toFixed(1),
    };
    chartData.push(paperBaseline);
  }

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '100px' }}>
        <Spin size="large" />
        <Paragraph style={{ marginTop: 20 }}>Loading experiments...</Paragraph>
      </div>
    );
  }

  const bestModel = modelResults.reduce(
    (best, current) =>
      current.metrics.f1 > (best?.metrics.f1 || 0) ? current : best,
    null as ModelResult | null
  );

  return (
    <div>
      <Title level={2}>Experimental Results</Title>
      <Paragraph>
        Compare model performance across different architectures and configurations.
      </Paragraph>

      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          closable
          style={{ marginBottom: 20 }}
        />
      )}

      {modelResults.length === 0 && !loading && (
        <Alert
          message="No Experiments Yet"
          description="Train models to see experimental results here. Run: python -m backend.training.train_roberta"
          type="info"
          style={{ marginBottom: 20 }}
        />
      )}

      {modelResults.length > 0 && (
        <>
          <Row gutter={[16, 16]} style={{ marginBottom: 30 }}>
            <Col span={8}>
              <Card>
                <Statistic
                  title="Total Experiments"
                  value={modelResults.length}
                  suffix="models"
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card>
                <Statistic
                  title="Best F1 Score"
                  value={bestModel ? (bestModel.metrics.f1 * 100).toFixed(2) : 0}
                  precision={2}
                  suffix="%"
                />
                {bestModel && (
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {bestModel.model.replace(/_/g, ' ')}
                  </Text>
                )}
              </Card>
            </Col>
            <Col span={8}>
              <Card>
                <Statistic
                  title="Paper Reference (Full EMFRD)"
                  value={97.8}
                  precision={1}
                  suffix="%"
                />
                <Text type="secondary" style={{ fontSize: 12 }}>
                  Reported accuracy
                </Text>
              </Card>
            </Col>
          </Row>

          <Card title="Model Performance Comparison" style={{ marginBottom: 30 }}>
            <Table
              dataSource={modelResults}
              columns={columns}
              rowKey="model"
              pagination={false}
            />
          </Card>

          <Card title="Metrics Visualization">
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Legend />
                <Bar dataKey="Accuracy" fill="#8884d8" />
                <Bar dataKey="Precision" fill="#82ca9d" />
                <Bar dataKey="Recall" fill="#ffc658" />
                <Bar dataKey="F1 Score" fill="#ff7c7c" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </>
      )}

      {paperReference && (
        <Card title="Paper Reference Results" style={{ marginTop: 30 }}>
          <Paragraph>
            <Text strong>Note:</Text> These are reference values reported in the paper.
            Our reproduction results are shown separately above.
          </Paragraph>

          <Row gutter={[16, 16]}>
            <Col span={8}>
              <Card size="small">
                <Statistic
                  title="RoBERTa Baseline"
                  value={
                    paperReference.roberta_baseline
                      ? (paperReference.roberta_baseline.accuracy * 100).toFixed(1)
                      : 0
                  }
                  suffix="%"
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card size="small">
                <Statistic
                  title="RoBERTa + Contrastive"
                  value={
                    paperReference.roberta_contrastive
                      ? (paperReference.roberta_contrastive.accuracy * 100).toFixed(1)
                      : 0
                  }
                  suffix="%"
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card size="small">
                <Statistic
                  title="Full EMFRD"
                  value={
                    paperReference.fusion_full
                      ? (paperReference.fusion_full.accuracy * 100).toFixed(1)
                      : 0
                  }
                  suffix="%"
                />
              </Card>
            </Col>
          </Row>
        </Card>
      )}
    </div>
  );
}

export default ExperimentsPage;
