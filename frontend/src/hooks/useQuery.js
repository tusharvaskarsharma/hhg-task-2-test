import { useState } from 'react';
import { executeQuery } from '../api/query';

export const useQuery = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  const submitQuery = async (queryText, language) => {
    if (!queryText || !queryText.trim()) return;
    
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const response = await executeQuery(queryText, language);
      setData(response);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setData(null);
    setError(null);
    setLoading(false);
  };

  return { submitQuery, loading, error, data, reset };
};
