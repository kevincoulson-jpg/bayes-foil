import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import styled from 'styled-components';
import { Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1rem;
  background: #1a1a1a;
  color: white;
  min-height: 100vh;
`;

const MainContent = styled.div`
  display: flex;
  flex-direction: row;
  width: 100vw;
  max-width: 1100px;
  justify-content: center;
  align-items: flex-start;
  gap: 1.2rem;
`;

const LeftColumn = styled.div`
  display: flex;
  flex-direction: column;
  width: 320px;
  height: 520px;
  gap: 0.7rem;
`;

const RightColumn = styled.div`
  display: flex;
  flex-direction: column;
  width: 520px;
  height: 520px;
  gap: 0.7rem;
`;

const ChartRow = styled.div`
  flex: 1 1 0;
  background: #2a2a2a;
  border-radius: 8px;
  padding: 0.5rem 0.5rem 0.2rem 0.5rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-height: 0;
`;

const SVGContainer = styled.div`
  width: 100%;
  height: 220px;
  background: #2a2a2a;
  border-radius: 8px;
  padding: 0.5rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
`;

const NacaLabel = styled.div`
  font-size: 1.2rem;
  font-weight: bold;
  margin-bottom: 0.2rem;
  letter-spacing: 2px;
`;

const Controls = styled.div`
  display: flex;
  gap: 0.7rem;
  margin-bottom: 0.7rem;
`;

const Button = styled.button`
  padding: 0.3rem 0.7rem;
  border: none;
  border-radius: 4px;
  background: #4a90e2;
  color: white;
  cursor: pointer;
  font-size: 1rem;
  transition: background 0.2s;

  &:hover {
    background: #357abd;
  }

  &:disabled {
    background: #666;
    cursor: not-allowed;
  }
`;

const YAxisLatex = styled.div`
  position: absolute;
  left: -38px;
  top: 50%;
  transform: translateY(-50%) rotate(-90deg);
  font-size: 1.1rem;
  color: #fff;
  pointer-events: none;
`;

const AirfoilVisualizer = () => {
  const [history, setHistory] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    fetch('http://localhost:5000/api/optimization-history')
      .then(res => res.json())
      .then(data => setHistory(data));
  }, []);

  useEffect(() => {
    let interval;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentIndex(prev => (prev + 1) % history.length);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isPlaying, history.length]);

  const currentAirfoil = history[currentIndex] || { points: [], parameters: {}, objective_value: 0 };
  const mArr = history.map(h => h.parameters.m);
  const pArr = history.map(h => h.parameters.p);
  const tArr = history.map(h => h.parameters.t);
  const clcdArr = history.map(h => h.objective_value);

  // NACA 4-digit label
  const nacaLabel = currentAirfoil.parameters
    ? `NACA${String(Math.round(currentAirfoil.parameters.m * 100))}${String(Math.round(currentAirfoil.parameters.p * 10))}${String(Math.round(currentAirfoil.parameters.t * 100)).padStart(2, '0')}`
    : '';

  // Chart.js scatter data for each parameter
  const makeScatterData = (arr, color, label) => ({
    labels: history.map((_, i) => i + 1),
    datasets: [
      {
        label,
        data: arr,
        backgroundColor: color,
        pointRadius: 3,
        pointHoverRadius: 5,
        showLine: false
      },
      {
        label: 'Current',
        data: arr.map((v, i) => i === currentIndex ? v : null),
        backgroundColor: '#ff4444',
        pointRadius: 7,
        pointHoverRadius: 9,
        showLine: false
      }
    ]
  });

  const scatterOptions = (yLabel, showXLabel = false) => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { mode: 'index', intersect: false }
    },
    animation: { duration: 0 },
    scales: {
      x: {
        grid: { color: 'rgba(255,255,255,0.1)' },
        ticks: { color: 'white', font: { size: 10 } },
        title: showXLabel ? { display: true, text: 'Iteration', color: 'white', font: { size: 12 } } : undefined
      },
      y: {
        grid: { color: 'rgba(255,255,255,0.1)' },
        ticks: { color: 'white', font: { size: 10 } },
        title: { display: true, text: yLabel, color: 'white', font: { size: 12 } }
      }
    }
  });

  const clcdScatterData = {
    labels: history.map((_, i) => i + 1),
    datasets: [
      {
        label: 'Cl/Cd',
        data: clcdArr,
        backgroundColor: '#4a90e2',
        pointRadius: 3,
        pointHoverRadius: 5,
        showLine: false
      },
      {
        label: 'Current',
        data: clcdArr.map((v, i) => i === currentIndex ? v : null),
        backgroundColor: '#ff4444',
        pointRadius: 7,
        pointHoverRadius: 9,
        showLine: false
      }
    ]
  };

  const clcdScatterOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { mode: 'index', intersect: false },
      title: { display: false }
    },
    animation: { duration: 0 },
    scales: {
      x: {
        grid: { color: 'rgba(255,255,255,0.1)' },
        ticks: { color: 'white', font: { size: 10 } },
        title: { display: true, text: 'Iteration', color: 'white', font: { size: 12 } }
      },
      y: {
        grid: { color: 'rgba(255,255,255,0.1)' },
        ticks: { color: 'white', font: { size: 10 } },
        title: { display: true, text: '', color: 'white', font: { size: 12 } },
        // We'll render LaTeX label below
      }
    }
  };

  const pointsToPath = (points) => {
    if (!points.length) return '';
    return points.map((point, i) => 
      `${i === 0 ? 'M' : 'L'} ${point[0] * 700 + 50} ${-point[1] * 700 + 80}`
    ).join(' ') + 'Z';
  };

  return (
    <Container>
      <h1 style={{ fontSize: '1.7rem', marginBottom: '0.7rem' }}>Airfoil Optimization Visualization</h1>
      <Controls>
        <Button onClick={() => setIsPlaying(!isPlaying)}>
          {isPlaying ? 'Pause' : 'Play'}
        </Button>
        <Button 
          onClick={() => setCurrentIndex(prev => (prev - 1 + history.length) % history.length)}
          disabled={isPlaying}
        >
          Previous
        </Button>
        <Button 
          onClick={() => setCurrentIndex(prev => (prev + 1) % history.length)}
          disabled={isPlaying}
        >
          Next
        </Button>
      </Controls>
      <MainContent>
        <LeftColumn>
          <ChartRow>
            <Scatter data={makeScatterData(mArr, '#4a90e2', 'm')} options={scatterOptions('Max Camber, m', false)} />
          </ChartRow>
          <ChartRow>
            <Scatter data={makeScatterData(pArr, '#4a90e2', 'p')} options={scatterOptions('Max Camber Location, p', false)} />
          </ChartRow>
          <ChartRow>
            <Scatter data={makeScatterData(tArr, '#4a90e2', 't')} options={scatterOptions('Thickness, t', true)} />
          </ChartRow>
        </LeftColumn>
        <RightColumn>
          <SVGContainer>
            <NacaLabel>{nacaLabel}</NacaLabel>
            <svg width="100%" height="160" viewBox="0 0 800 160">
              <motion.path
                d={pointsToPath(currentAirfoil.points)}
                fill="none"
                stroke="#4a90e2"
                strokeWidth="2"
                initial={false}
                animate={{ d: pointsToPath(currentAirfoil.points) }}
                transition={{ duration: 0.5 }}
              />
            </svg>
          </SVGContainer>
          <ChartRow style={{ height: 'calc(50% - 0.5rem)', position: 'relative' }}>
            <YAxisLatex>
              <BlockMath math={'C_l/C_d'} />
            </YAxisLatex>
            <Scatter data={clcdScatterData} options={clcdScatterOptions} />
          </ChartRow>
        </RightColumn>
      </MainContent>
    </Container>
  );
};

export default AirfoilVisualizer; 