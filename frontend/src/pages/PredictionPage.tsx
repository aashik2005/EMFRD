import { useState } from 'react';
import {
  Card,
  Input,
  Button,
  Typography,
  Alert,
  Spin,
  Row,
  Col,
  Progress,
  Statistic,
  Select,
} from 'antd';
import { api, PredictionResponse } from '../api/client';

const { Title, Paragraph, Text } = Typography;
const { TextArea } = Input;

function PredictionPage() {
  const [reviewText, setReviewText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('roberta_baseline');

  const handlePredict = async () => {
    if (!reviewText.trim()) {
      setError('Please enter a review text');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      let prediction: PredictionResponse;

      if (selectedModel === 'full_emfrd') {
        prediction = await api.predictFull({ review_text: reviewText });
      } else if (selectedModel === 'roberta_contrastive') {
        prediction = await api.predictContrastive({ review_text: reviewText });
      } else {
        prediction = await api.predictRoberta({ review_text: reviewText });
      }

      setResult(prediction);
    } catch (err: any) {
      setError(err.message || 'Prediction failed');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setReviewText('');
    setResult(null);
    setError(null);
  };

  const exampleReviews = [
    {
      label: 'Suspicious (likely fake)',
      text: 'This is the absolute BEST product ever!!! Amazing amazing amazing! Everyone should buy this immediately! Perfect in every way! 5 stars!!!!!',
    },
    {
      label: 'Genuine (likely real)',
      text: 'The product arrived on time and works as described. Battery life is decent, lasting about 6-7 hours with normal use. Build quality feels solid. Good value for the price.',
    },
  ];

  return (
    <div>
      <Title level={2}>Fake Review Detection</Title>
      <Paragraph>
        Enter a review to classify it as FAKE or GENUINE using the trained model.
      </Paragraph>

      <Row gutter={[16, 16]}>
        <Col span={12}>
          <Card title="Input Review">
            <Paragraph>Model:</Paragraph>
            <Select
              value={selectedModel}
              onChange={setSelectedModel}
              style={{ width: '100%', marginBottom: 16 }}
              options={[
                { value: 'roberta_baseline', label: 'RoBERTa Baseline' },
                { value: 'roberta_contrastive', label: 'RoBERTa + Contrastive Learning (Phase 2)' },
                { value: 'full_emfrd', label: 'Full EMFRD (when available)' },
              ]}
            />

            <Paragraph>Enter review text:</Paragraph>
            <TextArea
              rows={6}
              value={reviewText}
              onChange={(e) => setReviewText(e.target.value)}
              placeholder="Enter review text here..."
              style={{ marginBottom: 16 }}
            />

            <div style={{ marginBottom: 16 }}>
              <Text strong>Examples:</Text>
              {exampleReviews.map((example, idx) => (
                <div key={idx} style={{ marginTop: 8 }}>
                  <Button
                    size="small"
                    type="link"
                    onClick={() => setReviewText(example.text)}
                  >
                    {example.label}
                  </Button>
                </div>
              ))}
            </div>

            <Button
              type="primary"
              onClick={handlePredict}
              loading={loading}
              style={{ marginRight: 8 }}
              disabled={!reviewText.trim()}
            >
              Predict
            </Button>
            <Button onClick={handleClear}>Clear</Button>

            {error && (
              <Alert
                message="Error"
                description={error}
                type="error"
                closable
                style={{ marginTop: 16 }}
              />
            )}
          </Card>
        </Col>

        <Col span={12}>
          <Card title="Prediction Result">
            {loading && (
              <div style={{ textAlign: 'center', padding: 50 }}>
                <Spin size="large" />
                <Paragraph style={{ marginTop: 20 }}>Analyzing review...</Paragraph>
              </div>
            )}

            {!loading && !result && (
              <div style={{ textAlign: 'center', padding: 50, color: '#999' }}>
                <Paragraph>Enter a review and click "Predict" to see results</Paragraph>
              </div>
            )}

            {!loading && result && (
              <div>
                <Alert
                  message={
                    <Title level={3} style={{ margin: 0 }}>
                      {result.prediction}
                    </Title>
                  }
                  description={
                    <div>
                      <Paragraph>
                        This review is classified as{' '}
                        <Text strong>{result.prediction}</Text>
                      </Paragraph>
                      <Paragraph type="secondary">
                        Model: {result.model_used}
                      </Paragraph>
                    </div>
                  }
                  type={result.prediction === 'FAKE' ? 'error' : 'success'}
                  showIcon
                  style={{ marginBottom: 24 }}
                />

                <Row gutter={[16, 16]}>
                  <Col span={24}>
                    <Card size="small">
                      <Statistic
                        title="Confidence"
                        value={result.confidence * 100}
                        precision={2}
                        suffix="%"
                      />
                      <Progress
                        percent={result.confidence * 100}
                        strokeColor={
                          result.prediction === 'FAKE' ? '#ff4d4f' : '#52c41a'
                        }
                        showInfo={false}
                      />
                    </Card>
                  </Col>
                </Row>

                <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
                  <Col span={12}>
                    <Card size="small">
                      <Statistic
                        title="Fake Probability"
                        value={result.fake_probability * 100}
                        precision={2}
                        suffix="%"
                        valueStyle={{ color: '#ff4d4f' }}
                      />
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card size="small">
                      <Statistic
                        title="Genuine Probability"
                        value={result.genuine_probability * 100}
                        precision={2}
                        suffix="%"
                        valueStyle={{ color: '#52c41a' }}
                      />
                    </Card>
                  </Col>
                </Row>

                {result.semantic_confidence && (
                  <Card size="small" style={{ marginTop: 16 }} title="Component Breakdown">
                    <Row gutter={[16, 16]}>
                      <Col span={8}>
                        <Statistic
                          title="Semantic"
                          value={result.semantic_confidence * 100}
                          precision={1}
                          suffix="%"
                        />
                      </Col>
                      {result.graph_confidence && (
                        <Col span={8}>
                          <Statistic
                            title="Graph"
                            value={result.graph_confidence * 100}
                            precision={1}
                            suffix="%"
                          />
                        </Col>
                      )}
                      {result.adversarial_confidence && (
                        <Col span={8}>
                          <Statistic
                            title="Adversarial"
                            value={result.adversarial_confidence * 100}
                            precision={1}
                            suffix="%"
                          />
                        </Col>
                      )}
                    </Row>
                  </Card>
                )}
              </div>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
}

export default PredictionPage;
