-- Supabase SQL Schema for MMER Analysis History
-- Run this in Supabase SQL Editor: https://app.supabase.com/project/YOUR_PROJECT/sql

-- 1. Enable UUID extension (usually already enabled)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Create analysis_history table
CREATE TABLE IF NOT EXISTS public.analysis_history (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  modality VARCHAR(50) NOT NULL CHECK (modality IN ('facial', 'speech', 'multimodal')),
  emotion VARCHAR(50) NOT NULL,
  confidence FLOAT NOT NULL,
  probabilities JSONB,
  explainability VARCHAR(50),
  concordance VARCHAR(50),
  note TEXT,
  pinned BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_analysis_history_user_id 
  ON public.analysis_history(user_id);

CREATE INDEX IF NOT EXISTS idx_analysis_history_created_at 
  ON public.analysis_history(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_analysis_history_user_created 
  ON public.analysis_history(user_id, created_at DESC);

-- 4. Enable Row Level Security (RLS)
ALTER TABLE public.analysis_history ENABLE ROW LEVEL SECURITY;

-- 5. Create RLS policy: Users can only view their own records
CREATE POLICY "Users can view their own analysis history"
  ON public.analysis_history
  FOR SELECT
  USING (auth.uid() = user_id);

-- 6. Create RLS policy: Users can insert their own records
CREATE POLICY "Users can insert their own analysis history"
  ON public.analysis_history
  FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- 7. Create RLS policy: Users can update their own records
CREATE POLICY "Users can update their own analysis history"
  ON public.analysis_history
  FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- 8. Create RLS policy: Users can delete their own records
CREATE POLICY "Users can delete their own analysis history"
  ON public.analysis_history
  FOR DELETE
  USING (auth.uid() = user_id);

-- 9. Create trigger for updated_at timestamp
CREATE OR REPLACE FUNCTION public.update_analysis_history_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = CURRENT_TIMESTAMP;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_analysis_history_timestamp_trigger
  BEFORE UPDATE ON public.analysis_history
  FOR EACH ROW
  EXECUTE FUNCTION public.update_analysis_history_timestamp();

-- 10. Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON public.analysis_history TO authenticated;
