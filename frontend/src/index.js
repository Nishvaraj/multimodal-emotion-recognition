/*
  Frontend entrypoint for the Multi-Modal Emotion Recognition web application.

  This module mounts the root React tree and keeps startup wiring minimal so
  routing, state, and analytics orchestration remain centralized in App.
*/

// --- Imports ---
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// --- Application Bootstrap ---
// Mount the React application into the root DOM node.
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
