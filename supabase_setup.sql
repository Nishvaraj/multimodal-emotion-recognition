-- Supabase Database Schema for Emotion Recognition Logging
-- Run this SQL in your Supabase SQL Editor to create the necessary tables

-- Create emotion predictions table
CREATE TABLE IF NOT EXISTS emotion_predictions (
    id BIGSERIAL PRIMARY KEY,
    prediction_type VARCHAR(50) NOT NULL, -- 'facial', 'speech', or 'combined'
    emotion VARCHAR(50),
    confidence FLOAT,
    all_probabilities JSONB, -- Store all emotion probabilities as JSON
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    device VARCHAR(50), -- 'cuda' or 'cpu'
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL, -- Optional: link to authenticated users
    session_id VARCHAR(255), -- Optional: track user sessions
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON emotion_predictions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_type ON emotion_predictions(prediction_type);
CREATE INDEX IF NOT EXISTS idx_predictions_emotion ON emotion_predictions(emotion);
CREATE INDEX IF NOT EXISTS idx_predictions_user ON emotion_predictions(user_id);

-- Enable Row Level Security (RLS)
ALTER TABLE emotion_predictions ENABLE ROW LEVEL SECURITY;

-- Create policy to allow all inserts (for anonymous usage)
CREATE POLICY "Allow all inserts" ON emotion_predictions
    FOR INSERT
    WITH CHECK (true);

-- Create policy to allow users to view their own predictions
CREATE POLICY "Users can view own predictions" ON emotion_predictions
    FOR SELECT
    USING (auth.uid() = user_id OR user_id IS NULL);

-- Optional: Create a view for aggregated statistics
CREATE OR REPLACE VIEW emotion_statistics AS
SELECT 
    prediction_type,
    emotion,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence,
    DATE_TRUNC('day', timestamp) as date
FROM emotion_predictions
GROUP BY prediction_type, emotion, DATE_TRUNC('day', timestamp)
ORDER BY date DESC, count DESC;

-- Grant access to the view
GRANT SELECT ON emotion_statistics TO anon, authenticated;

-- Optional: Create function to get recent predictions
CREATE OR REPLACE FUNCTION get_recent_predictions(
    limit_count INTEGER DEFAULT 100,
    pred_type VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    prediction_type VARCHAR,
    emotion VARCHAR,
    confidence FLOAT,
    timestamp TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ep.id,
        ep.prediction_type,
        ep.emotion,
        ep.confidence,
        ep.timestamp
    FROM emotion_predictions ep
    WHERE (pred_type IS NULL OR ep.prediction_type = pred_type)
    ORDER BY ep.timestamp DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission on the function
GRANT EXECUTE ON FUNCTION get_recent_predictions TO anon, authenticated;

-- Create function to get emotion distribution
CREATE OR REPLACE FUNCTION get_emotion_distribution(
    pred_type VARCHAR DEFAULT NULL,
    days_back INTEGER DEFAULT 7
)
RETURNS TABLE (
    emotion VARCHAR,
    count BIGINT,
    percentage NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    WITH counts AS (
        SELECT 
            ep.emotion,
            COUNT(*) as cnt
        FROM emotion_predictions ep
        WHERE 
            (pred_type IS NULL OR ep.prediction_type = pred_type)
            AND ep.timestamp > NOW() - INTERVAL '1 day' * days_back
        GROUP BY ep.emotion
    ),
    total AS (
        SELECT SUM(cnt) as total_count FROM counts
    )
    SELECT 
        c.emotion,
        c.cnt as count,
        ROUND((c.cnt::NUMERIC / t.total_count::NUMERIC) * 100, 2) as percentage
    FROM counts c, total t
    ORDER BY c.cnt DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION get_emotion_distribution TO anon, authenticated;

-- Insert a test record (optional)
INSERT INTO emotion_predictions (prediction_type, emotion, confidence, all_probabilities, device)
VALUES ('facial', 'happy', 0.95, '{"happy": 0.95, "neutral": 0.03, "sad": 0.02}'::jsonb, 'cpu');

-- Display setup completion message
DO $$
BEGIN
    RAISE NOTICE '✅ Supabase schema setup complete!';
    RAISE NOTICE 'Table created: emotion_predictions';
    RAISE NOTICE 'Views created: emotion_statistics';
    RAISE NOTICE 'Functions created: get_recent_predictions, get_emotion_distribution';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '1. Copy your SUPABASE_URL and SUPABASE_KEY from Project Settings > API';
    RAISE NOTICE '2. Add them to your .env file';
    RAISE NOTICE '3. Install supabase-py: pip install supabase';
END $$;
