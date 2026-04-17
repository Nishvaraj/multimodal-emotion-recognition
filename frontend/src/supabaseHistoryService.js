import { supabase } from './supabaseClient';

// ========== SUPABASE HISTORY OPERATIONS ==========

export async function saveAnalysisToSupabase(record) {
  try {
    const { data: { session } } = await supabase.auth.getSession();
    if (!session?.user?.id) {
      console.error('No authenticated user');
      return false;
    }

    const scoreFromRecord = Number(record.concordance_score);
    const hasConcordanceScore = Number.isFinite(scoreFromRecord);

    const probabilitiesPayload = {
      ...(record.probabilities || {})
    };

    // Keep a JSON fallback so score survives even if DB schema is older.
    if (hasConcordanceScore) {
      probabilitiesPayload.__concordance_score = scoreFromRecord;
    }

    const insertPayload = {
      user_id: session.user.id,
      modality: record.modality,
      emotion: record.emotion,
      confidence: record.confidence,
      probabilities: probabilitiesPayload,
      explainability: record.explainability,
      concordance: record.concordance,
      note: record.note || '',
      pinned: record.pinned || false,
      created_at: record.createdAt
    };

    if (hasConcordanceScore) {
      insertPayload.concordance_score = scoreFromRecord;
    }

    let { error } = await supabase
      .from('analysis_history')
      .insert([insertPayload]);

    // Backward-compat fallback for databases without concordance_score column.
    if (error && hasConcordanceScore && String(error.message || '').toLowerCase().includes('concordance_score')) {
      const fallbackPayload = { ...insertPayload };
      delete fallbackPayload.concordance_score;
      ({ error } = await supabase
        .from('analysis_history')
        .insert([fallbackPayload]));
    }

    if (error) {
      console.error('Error saving to Supabase:', error);
      return false;
    }
    console.log('✓ Analysis saved to Supabase');
    return true;
  } catch (err) {
    console.error('Error in saveAnalysisToSupabase:', err);
    return false;
  }
}

export async function loadAnalysisHistoryFromSupabase(dateFilter = null) {
  try {
    const { data: { session } } = await supabase.auth.getSession();
    if (!session?.user?.id) {
      console.error('No authenticated user');
      return [];
    }

    let query = supabase
      .from('analysis_history')
      .select('*')
      .eq('user_id', session.user.id)
      .order('created_at', { ascending: false });

    if (dateFilter) {
      // Filter by date if provided (YYYY-MM-DD format)
      const startOfDay = `${dateFilter}T00:00:00`;
      const endOfDay = `${dateFilter}T23:59:59`;
      query = query
        .gte('created_at', startOfDay)
        .lte('created_at', endOfDay);
    }

    const { data, error } = await query;

    if (error) {
      console.error('Error loading history:', error);
      return [];
    }

    // Transform Supabase data to frontend format
    return (data || []).map(record => ({
      id: record.id,
      modality: record.modality,
      emotion: record.emotion,
      confidence: record.confidence,
      probabilities: record.probabilities || {},
      explainability: record.explainability,
      concordance: record.concordance,
      concordance_score: Number.isFinite(Number(record.concordance_score))
        ? Number(record.concordance_score)
        : Number(record?.probabilities?.__concordance_score),
      note: record.note || '',
      pinned: record.pinned || false,
      createdAt: record.created_at
    }));
  } catch (err) {
    console.error('Error in loadAnalysisHistoryFromSupabase:', err);
    return [];
  }
}

export async function updateAnalysisNote(recordId, note) {
  try {
    const { error } = await supabase
      .from('analysis_history')
      .update({ note })
      .eq('id', recordId);

    if (error) {
      console.error('Error updating note:', error);
      return false;
    }
    return true;
  } catch (err) {
    console.error('Error in updateAnalysisNote:', err);
    return false;
  }
}

export async function toggleAnalysisPin(recordId, pinned) {
  try {
    const { error } = await supabase
      .from('analysis_history')
      .update({ pinned: !pinned })
      .eq('id', recordId);

    if (error) {
      console.error('Error toggling pin:', error);
      return false;
    }
    return true;
  } catch (err) {
    console.error('Error in toggleAnalysisPin:', err);
    return false;
  }
}

export async function deleteAnalysisRecord(recordId) {
  try {
    const { error } = await supabase
      .from('analysis_history')
      .delete()
      .eq('id', recordId);

    if (error) {
      console.error('Error deleting record:', error);
      return false;
    }
    return true;
  } catch (err) {
    console.error('Error in deleteAnalysisRecord:', err);
    return false;
  }
}

// ========== REAL-TIME SUBSCRIPTION ==========

export function subscribeToAnalysisHistory(callback) {
  try {
    const subscription = supabase
      .channel('analysis_history:changes')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'analysis_history'
        },
        (payload) => {
          console.log('Real-time update:', payload);
          callback(payload);
        }
      )
      .subscribe();

    return subscription;
  } catch (err) {
    console.error('Error subscribing to history:', err);
    return null;
  }
}
