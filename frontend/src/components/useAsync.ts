import { useCallback, useEffect, useRef, useState } from "react";
import { errorMessage } from "../api/client";

/** Load data with loading / error state; `reload` re-runs the loader. */
export function useAsync<T>(loader: () => Promise<T>, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const seq = useRef(0);

  const run = useCallback(
    async (silent = false) => {
      const id = ++seq.current;
      if (!silent) setLoading(true);
      try {
        const result = await loader();
        if (id === seq.current) {
          setData(result);
          setError(null);
        }
      } catch (e) {
        if (id === seq.current) setError(errorMessage(e));
      } finally {
        if (id === seq.current) setLoading(false);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    deps,
  );

  useEffect(() => {
    run();
  }, [run]);

  return { data, error, loading, reload: run, setData };
}
