"""
Database service for session storage and prediction history.
Uses SQLite to store sessions and predictions locally.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import csv
from io import StringIO
import numpy as np

class SessionDatabase:
    """Manage session storage using SQLite"""
    
    def __init__(self, db_path: str = "data/sessions.db"):
        """
        Initialize database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                user_name TEXT,
                total_predictions INTEGER DEFAULT 0,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                modality TEXT,
                emotion TEXT,
                confidence REAL,
                probabilities TEXT,
                grad_cam_image TEXT,
                saliency_map TEXT,
                input_file_path TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        
        # Concordance records table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concordance_records (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                facial_emotion TEXT,
                facial_confidence REAL,
                speech_emotion TEXT,
                speech_confidence REAL,
                concordance_status TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_session(self, user_id: str = None, user_name: str = None, notes: str = None) -> str:
        """
        Create a new session
        
        Args:
            user_id: Optional user identifier
            user_name: Optional user name
            notes: Optional session notes
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sessions (id, user_id, user_name, notes)
            VALUES (?, ?, ?, ?)
        """, (session_id, user_id, user_name, notes))
        
        conn.commit()
        conn.close()
        
        return session_id
    
    def save_prediction(
        self,
        session_id: str,
        modality: str,
        emotion: str,
        confidence: float,
        probabilities: Dict = None,
        grad_cam_image: str = None,
        saliency_map: str = None,
        input_file_path: str = None
    ) -> str:
        """
        Save a prediction to a session
        
        Args:
            session_id: Session ID
            modality: 'facial', 'speech', or 'combined'
            emotion: Predicted emotion
            confidence: Confidence score (0-1)
            probabilities: Dict of emotion probabilities
            grad_cam_image: Base64 encoded Grad-CAM image (for facial)
            saliency_map: Base64 encoded saliency map (for speech)
            input_file_path: Path to input file (optional)
            
        Returns:
            Prediction ID
        """
        prediction_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert probabilities dict to JSON
        probs_json = json.dumps(probabilities or {})
        
        cursor.execute("""
            INSERT INTO predictions
            (id, session_id, modality, emotion, confidence, probabilities, 
             grad_cam_image, saliency_map, input_file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction_id, session_id, modality, emotion, confidence,
            probs_json, grad_cam_image, saliency_map, input_file_path
        ))
        
        # Update session's prediction count and timestamp
        cursor.execute("""
            UPDATE sessions 
            SET total_predictions = total_predictions + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (session_id,))
        
        conn.commit()
        conn.close()
        
        return prediction_id
    
    def save_concordance(
        self,
        session_id: str,
        facial_emotion: str,
        facial_confidence: float,
        speech_emotion: str,
        speech_confidence: float,
        concordance_status: str  # "MATCH" or "MISMATCH"
    ) -> str:
        """
        Save a concordance record
        
        Args:
            session_id: Session ID
            facial_emotion: Facial emotion
            facial_confidence: Facial confidence
            speech_emotion: Speech emotion
            speech_confidence: Speech confidence
            concordance_status: "MATCH" or "MISMATCH"
            
        Returns:
            Concordance record ID
        """
        record_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO concordance_records
            (id, session_id, facial_emotion, facial_confidence, 
             speech_emotion, speech_confidence, concordance_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            record_id, session_id, facial_emotion, facial_confidence,
            speech_emotion, speech_confidence, concordance_status
        ))
        
        conn.commit()
        conn.close()
        
        return record_id
    
    def get_session(self, session_id: str) -> Dict:
        """
        Get session details
        
        Args:
            session_id: Session ID
            
        Returns:
            Session dict or None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        session = cursor.fetchone()
        
        conn.close()
        
        return dict(session) if session else None
    
    def get_all_sessions(self, limit: int = 100) -> List[Dict]:
        """
        Get all sessions
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session dicts
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM sessions 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        sessions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return sessions
    
    def get_session_predictions(self, session_id: str) -> List[Dict]:
        """
        Get all predictions for a session
        
        Args:
            session_id: Session ID
            
        Returns:
            List of prediction dicts
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM predictions 
            WHERE session_id = ?
            ORDER BY timestamp DESC
        """, (session_id,))
        
        predictions = []
        for row in cursor.fetchall():
            pred = dict(row)
            pred['probabilities'] = json.loads(pred['probabilities'] or '{}')
            predictions.append(pred)
        
        conn.close()
        
        return predictions
    
    def get_session_concordance(self, session_id: str) -> List[Dict]:
        """
        Get all concordance records for a session
        
        Args:
            session_id: Session ID
            
        Returns:
            List of concordance dicts
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM concordance_records 
            WHERE session_id = ?
            ORDER BY timestamp DESC
        """, (session_id,))
        
        records = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return records
    
    def export_session_csv(self, session_id: str) -> str:
        """
        Export session predictions as CSV
        
        Args:
            session_id: Session ID
            
        Returns:
            CSV string
        """
        session = self.get_session(session_id)
        predictions = self.get_session_predictions(session_id)
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            'Session ID', 'User', 'Created At',
            'Prediction #', 'Modality', 'Emotion', 'Confidence', 'Timestamp'
        ])
        
        # Predictions
        for i, pred in enumerate(predictions, 1):
            writer.writerow([
                session_id,
                session.get('user_name') or 'Anonymous',
                session.get('created_at'),
                i,
                pred['modality'],
                pred['emotion'],
                f"{pred['confidence']:.2%}",
                pred['timestamp']
            ])
        
        return output.getvalue()
    
    def export_session_json(self, session_id: str) -> str:
        """
        Export session as JSON
        
        Args:
            session_id: Session ID
            
        Returns:
            JSON string
        """
        session = self.get_session(session_id)
        predictions = self.get_session_predictions(session_id)
        concordance = self.get_session_concordance(session_id)
        
        export_data = {
            "session": {
                "id": session['id'],
                "user_name": session.get('user_name'),
                "user_id": session.get('user_id'),
                "notes": session.get('notes'),
                "created_at": session['created_at'],
                "updated_at": session['updated_at'],
                "total_predictions": session['total_predictions']
            },
            "predictions": predictions,
            "concordance_records": concordance
        }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its predictions
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete predictions first (due to foreign key)
        cursor.execute("DELETE FROM predictions WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM concordance_records WHERE session_id = ?", (session_id,))
        
        # Delete session
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        
        conn.commit()
        conn.close()
        
        return True
    
    def get_statistics(self, session_id: str) -> Dict:
        """
        Get session statistics
        
        Args:
            session_id: Session ID
            
        Returns:
            Statistics dict
        """
        predictions = self.get_session_predictions(session_id)
        concordance = self.get_session_concordance(session_id)
        
        # Emotion frequency
        emotion_freq = {}
        modality_freq = {}
        
        for pred in predictions:
            emotion = pred['emotion']
            modality = pred['modality']
            emotion_freq[emotion] = emotion_freq.get(emotion, 0) + 1
            modality_freq[modality] = modality_freq.get(modality, 0) + 1
        
        # Concordance stats
        matches = sum(1 for c in concordance if c['concordance_status'] == 'MATCH')
        mismatches = sum(1 for c in concordance if c['concordance_status'] == 'MISMATCH')
        
        avg_confidence = np.mean([p['confidence'] for p in predictions]) if predictions else 0
        
        return {
            "total_predictions": len(predictions),
            "emotion_frequency": emotion_freq,
            "modality_frequency": modality_freq,
            "concordance_matches": matches,
            "concordance_mismatches": mismatches,
            "average_confidence": float(avg_confidence),
            "session_duration": len([p for p in predictions])
        }


# Global database instance
db = SessionDatabase()
