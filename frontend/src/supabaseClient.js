import { createClient } from '@supabase/supabase-js';

// Read the Supabase connection settings from the React environment.
const supabaseUrl = process.env.REACT_APP_SUPABASE_URL;
const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables');
}

// Shared Supabase client used across the frontend services.
export const supabase = createClient(supabaseUrl, supabaseAnonKey);
