import { useEffect, useState } from 'react';
import { Card, Row, Col, Statistic, Typography, Spin, Alert, Table } from 'antd';
import { CheckCircleOutlined, ExperimentOutlined } from '@ant-design/icons';
import { api, MetricsData } from '../api/client';

const { Title, Paragraph, Text } = Typography;

interface ModelStatus {
  name: string;
  displayName: string;
  trained: boolean;
  metrics?: MetricsData;
}

function DashboardPage() {
  const [loading, setLoading] = useState(true);
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [paperReference, setPaperReference] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);

      // Load available models
      const modelsData = await api.getAvailableModels();

      // Load paper reference
      const paperData = await api.getPaperReference();
      setPaperReference(paperData);

      // Format model data
      const modelStatus: ModelStatus[] = [
        {
          name: 'roberta_baseline',
          displayName: 'RoBERTa Baseline',
          trained: false,
        },
        {
          name: 'roberta_contrastive',
          displayName: 'RoBERTa + Contrastive',
          trained: false,
        },
        {
          name: 'hgnn',
          displayName: 'HGNN',
          trained: false,
        },
        {
          name: 'gan',
          displayName: 'GAN Adversarial',
          trained: false,
        },
        {
          name: 'fusion',
          displayName: 'Full EMFRD',
          trained: false,
        },
      ];

      // Update with actual trained status
      modelsData.models.forEach((model: any) => {
        const modelStatus = modelStatus.find((m) => m.name === model.name);
        if (modelStatus) {
          modelStatus.trained = model.trained;
          modelStatus.metrics = model.metrics;
        }
      });

      setModels(modelStatus);
      setError(null);
    } catch (err: any) {
      setError(err.message || 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const columns = [
    {
      title: 'Model',
      dataIndex: 'displayName',
      key: 'displayName',
    },
    {
      title: 'Status',
      dataIndex: 'trained',
      key: 'trained',
      render: (trained: boolean) => (
        <Text type={trained ? 'success' : 'secondary'}>
          {trained ? '✓ Trained' : '○ Not Trained'}
        </Text>
      ),
    },
    {
      title: 'Accuracy',
      key: 'accuracy',
      render: (_: any, record: ModelStatus) => {
        if (!record.trained || !record.metrics) return '-';
        return `${(record.metrics.accuracy * 100).toFixed(2)}%`;
      },
    },
    {
      title: 'F1 Score',
      key: 'f1',
      render: (_: any, record: ModelStatus) => {
        if (!record.trained || !record.metrics) return '-';
        return `${(record.metrics.f1 * 100).toFixed(2)}%`;
      },
    },
  ];

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '100px' }}>
        <Spin size="large" />
        <Paragraph style={{ marginTop: 20 }}>Loading dashboard...</Paragraph>
      </div>
    );
  }

  const trainedCount = models.filter((m) => m.trained).length;

  return (
    <div>
      <Title level={2}>EMFRD Dashboard</Title>
      <Paragraph>
        Explainable Multimodal Framework for Fake Review Detection Using Semantic,
        Graph, and Adversarial Learning
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

      <Row gutter={[16, 16]} style={{ marginTop: 30 }}>
        <Col span={8}>
          <Card>
            <Statistic
              title="Models Trained"
              value={trainedCount}
              suffix={`/ ${models.length}`}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="Paper Reference Accuracy"
              value={97.8}
              precision={1}
              suffix="%"
              prefix={<ExperimentOutlined />}
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              Full EMFRD (from paper)
            </Text>
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="Best Achieved Accuracy"
              value={
                models
                  .filter((m) => m.trained && m.metrics)
                  .reduce((max, m) => Math.max(max, m.metrics!.accuracy * 100), 0)
              }
              precision={2}
              suffix="%"
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              Our reproduction
            </Text>
          </Card>
        </Col>
      </Row>

      <Card title="Model Status" style={{ marginTop: 30 }}>
        <Table
          dataSource={models}
          columns={columns}
          rowKey="name"
          pagination={false}
        />
      </Card>

      <Card title="Architecture Overview" style={{ marginTop: 30 }}>
        <Paragraph>
          <Text strong>Phase 1 (Current): </Text>
          RoBERTa Baseline - Foundation semantic model
        </Paragraph>
        <Paragraph>
          <Text strong>Phase 2: </Text>
          RoBERTa + Contrastive Learning - Enhanced semantic representation
        </Paragraph>
        <Paragraph>
          <Text strong>Phase 3: </Text>
          HGNN - Graph-based behavioral analysis
        </Paragraph>
        <Paragraph>
          <Text strong>Phase 4: </Text>
          GAN Adversarial Training - Robustness against AI-generated reviews
        </Paragraph>
        <Paragraph>
          <Text strong>Phase 5: </Text>
          Gated Multimodal Fusion - Combine all modalities
        </Paragraph>
        <Paragraph>
          <Text strong>Phase 6: </Text>
          SHAP + Counterfactual Explainability
        </Paragraph>
      </Card>
    </div>
  );
}

export default DashboardPage;
