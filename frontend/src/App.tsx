import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { Layout, Menu, Typography } from 'antd';
import {
  HomeOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  DatabaseOutlined,
} from '@ant-design/icons';
import DashboardPage from './pages/DashboardPage';
import PredictionPage from './pages/PredictionPage';
import ExperimentsPage from './pages/ExperimentsPage';

const { Header, Content, Footer } = Layout;
const { Title } = Typography;

function App() {
  return (
    <BrowserRouter>
      <Layout style={{ minHeight: '100vh' }}>
        <Header style={{ display: 'flex', alignItems: 'center', padding: '0 50px' }}>
          <Title level={3} style={{ color: 'white', margin: '16px 30px 16px 0' }}>
            EMFRD
          </Title>
          <Menu
            theme="dark"
            mode="horizontal"
            defaultSelectedKeys={['dashboard']}
            style={{ flex: 1, minWidth: 0 }}
            items={[
              {
                key: 'dashboard',
                icon: <HomeOutlined />,
                label: <Link to="/">Dashboard</Link>,
              },
              {
                key: 'prediction',
                icon: <ExperimentOutlined />,
                label: <Link to="/prediction">Prediction</Link>,
              },
              {
                key: 'experiments',
                icon: <LineChartOutlined />,
                label: <Link to="/experiments">Experiments</Link>,
              },
            ]}
          />
        </Header>

        <Content style={{ padding: '50px' }}>
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/prediction" element={<PredictionPage />} />
            <Route path="/experiments" element={<ExperimentsPage />} />
          </Routes>
        </Content>

        <Footer style={{ textAlign: 'center' }}>
          EMFRD - Explainable Multimodal Framework for Fake Review Detection © 2024
        </Footer>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
