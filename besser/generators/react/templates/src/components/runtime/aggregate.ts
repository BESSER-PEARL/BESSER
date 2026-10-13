// Reduces bound records to one number (a metric card's value, a chart bar).
export type Aggregation = "sum" | "avg" | "count" | "min" | "max" | "median" | "first" | "last";

const ALIASES: Record<string, Aggregation> = {
  sum: "sum",
  avg: "avg",
  average: "avg",
  mean: "avg",
  count: "count",
  min: "min",
  minimum: "min",
  max: "max",
  maximum: "max",
  median: "median",
  first: "first",
  last: "last",
};

export const normalizeAggregation = (value?: string | null): Aggregation | undefined =>
  value ? ALIASES[String(value).trim().toLowerCase()] : undefined;

export const readField = (row: any, path?: string): any => {
  if (!row || !path) return undefined;
  return path.split(".").reduce((value: any, key) => {
    const current = Array.isArray(value) ? value[0] : value;
    return current == null ? undefined : current[key];
  }, row);
};

export const aggregate = (rows: any[], aggregation: Aggregation, field?: string): number => {
  if (aggregation === "count") return rows.length;
  const values = rows
    .map((row) => Number(readField(row, field)))
    .filter((value) => Number.isFinite(value));
  if (values.length === 0) return 0;
  switch (aggregation) {
    case "sum":
      return values.reduce((total, value) => total + value, 0);
    case "avg":
      return values.reduce((total, value) => total + value, 0) / values.length;
    case "min":
      return Math.min(...values);
    case "max":
      return Math.max(...values);
    case "median": {
      const sorted = [...values].sort((a, b) => a - b);
      const middle = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
    }
    case "first":
      return values[0];
    default:
      return values[values.length - 1];
  }
};
