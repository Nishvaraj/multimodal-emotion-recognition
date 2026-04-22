/*
  Supabase client bootstrap for the Multi-Modal Emotion Recognition frontend.

  This module centralizes database/auth client construction so all feature modules
  share one validated connection surface.
*/

// --- Imports ---
import { createClient } from '@supabase/supabase-js';

// --- Environment Configuration ---
// Read the Supabase connection settings from the React environment.
const supabaseUrl = process.env.REACT_APP_SUPABASE_URL;
const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables');
}

// --- Client Construction ---
// Shared Supabase client used across the frontend services.
export const supabase = createClient(supabaseUrl, supabaseAnonKey);
