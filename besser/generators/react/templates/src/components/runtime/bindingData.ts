// The data side of the runtime blocks: nested field values, filter expressions,
// the rows of a response and the URL of an endpoint.

export const isNestedField = (field?: string): boolean => !!field && field.includes(".");

export const getNestedValue = (obj: any, path: string): any => {
  if (!path || !obj) return undefined;
  if (!path.includes(".")) return obj[path];
  const parts = path.split(".");
  let value = obj;
  for (const part of parts) {
    if (value == null) return undefined;
    if (Array.isArray(value)) value = value[0];
    if (value == null) return undefined;
    value = value[part];
  }
  return value;
};

type FilterOperator = "eq" | "neq" | "gt" | "gte" | "lt" | "lte" | "contains" | "in" | "notin";

interface FilterCondition {
  field: string;
  op: FilterOperator;
  value: any;
}

export interface ParsedFilter {
  conditions: FilterCondition[];
  hasNestedFields: boolean;
  serverCompatible: boolean;
  queryParams: Record<string, string>;
}

const normalizeFilterInput = (filter: unknown): unknown => {
  if (filter === null || filter === undefined) return null;
  if (typeof filter === "string") {
    const trimmed = filter.trim();
    return trimmed.length > 0 ? trimmed : null;
  }
  if (Array.isArray(filter) || typeof filter === "object") return filter;
  return String(filter);
};

const parseScalar = (value: string): any => {
  const trimmed = value.trim();
  if (trimmed.length === 0) return "";
  if ((trimmed.startsWith("'") && trimmed.endsWith("'")) || (trimmed.startsWith("\"") && trimmed.endsWith("\""))) {
    return trimmed.slice(1, -1);
  }
  if (/^(true|false)$/i.test(trimmed)) return trimmed.toLowerCase() === "true";
  if (/^-?\d+(\.\d+)?$/.test(trimmed)) return Number(trimmed);
  if (/^null$/i.test(trimmed)) return null;
  return trimmed;
};

const parseList = (value: string): any[] => {
  const trimmed = value.trim();
  const inner =
    (trimmed.startsWith("(") && trimmed.endsWith(")")) || (trimmed.startsWith("[") && trimmed.endsWith("]"))
      ? trimmed.slice(1, -1)
      : trimmed;
  if (!inner.trim()) return [];
  return inner.split(",").map((item) => parseScalar(item));
};

const normalizeOperator = (op: string): FilterOperator => {
  const normalized = op.trim().toLowerCase();
  if (normalized === "==" || normalized === "=") return "eq";
  if (normalized === "!=") return "neq";
  if (normalized === ">") return "gt";
  if (normalized === ">=") return "gte";
  if (normalized === "<") return "lt";
  if (normalized === "<=") return "lte";
  if (normalized === "~" || normalized === "~=" || normalized === "contains") return "contains";
  if (normalized === "in") return "in";
  if (normalized === "not in") return "notin";
  return "eq";
};

const parseFilterObject = (obj: any): FilterCondition[] => {
  if (!obj) return [];
  if (Array.isArray(obj)) {
    return obj.flatMap((entry) => parseFilterObject(entry));
  }
  if (typeof obj !== "object") return [];
  if ("field" in obj && "value" in obj) {
    const op = normalizeOperator(obj.op || obj.operator || "=");
    const value = obj.value;
    return [{ field: String(obj.field), op, value }];
  }
  return Object.entries(obj).map(([key, value]) => {
    const op = Array.isArray(value) ? "in" : "eq";
    return { field: key, op, value } as FilterCondition;
  });
};

const isLikelyQueryString = (raw: string): boolean => {
  if (!raw.includes("=")) return false;
  if (/==|!=|>=|<=|>|<|~/.test(raw)) return false;
  if (/\s/.test(raw)) return false;
  return true;
};

const parseQueryString = (raw: string): FilterCondition[] => {
  const query = raw.startsWith("?") ? raw.slice(1) : raw;
  const params = new URLSearchParams(query);
  const valuesMap: Record<string, string[]> = {};
  params.forEach((value, key) => {
    if (!valuesMap[key]) valuesMap[key] = [];
    valuesMap[key].push(value);
  });
  return Object.entries(valuesMap).map(([field, values]) => {
    if (values.length > 1) {
      return { field, op: "in", value: values.map(parseScalar) };
    }
    return { field, op: "eq", value: parseScalar(values[0]) };
  });
};

const parseConditionString = (expr: string): FilterCondition | null => {
  const trimmed = expr.trim();
  if (!trimmed) return null;

  const inMatch = trimmed.match(/^(.+?)\s+(not\s+in|in)\s+(.+)$/i);
  if (inMatch) {
    const field = inMatch[1].trim();
    const op = normalizeOperator(inMatch[2]);
    const value = parseList(inMatch[3]);
    return { field, op, value };
  }

  const containsMatch = trimmed.match(/^(.+?)\s+contains\s+(.+)$/i);
  if (containsMatch) {
    const field = containsMatch[1].trim();
    const value = parseScalar(containsMatch[2]);
    return { field, op: "contains", value };
  }

  const opMatch = trimmed.match(/^(.+?)\s*(==|=|!=|>=|<=|>|<|~=|~|:)\s*(.+)$/);
  if (opMatch) {
    const field = opMatch[1].trim();
    const op = normalizeOperator(opMatch[2] === ":" ? "=" : opMatch[2]);
    const value = parseScalar(opMatch[3]);
    return { field, op, value };
  }

  return null;
};

const parseExpressionConditions = (raw: string): FilterCondition[] => {
  const parts = raw
    .split(/&&|\band\b|;|,/i)
    .map((part) => part.trim())
    .filter(Boolean);
  const conditions: FilterCondition[] = [];
  parts.forEach((part) => {
    const condition = parseConditionString(part);
    if (condition) conditions.push(condition);
  });
  return conditions;
};

const buildQueryParams = (conditions: FilterCondition[]): Record<string, string> => {
  const params: Record<string, string> = {};
  conditions.forEach((condition) => {
    if (!condition.field) return;
    const value = condition.value;
    if (value === undefined || value === null) return;
    if (Array.isArray(value)) return;
    params[condition.field] = String(value);
  });
  return params;
};

export const parseFilterExpression = (filter: unknown): ParsedFilter | null => {
  const normalized = normalizeFilterInput(filter);
  if (!normalized) return null;

  let conditions: FilterCondition[] = [];
  if (typeof normalized === "string") {
    const raw = normalized.trim();
    if (!raw) return null;
    if ((raw.startsWith("{") && raw.endsWith("}")) || (raw.startsWith("[") && raw.endsWith("]"))) {
      try {
        const parsed = JSON.parse(raw);
        conditions = parseFilterObject(parsed);
      } catch {
        conditions = [];
      }
    }
    if (!conditions.length && isLikelyQueryString(raw)) {
      conditions = parseQueryString(raw);
    }
    if (!conditions.length) {
      conditions = parseExpressionConditions(raw);
    }
  } else {
    conditions = parseFilterObject(normalized);
  }

  conditions = conditions.filter((cond) => !!cond.field);
  if (!conditions.length) return null;

  const hasNestedFields = conditions.some((cond) => isNestedField(cond.field));
  const serverCompatible =
    conditions.every((cond) => cond.op === "eq") &&
    !hasNestedFields &&
    conditions.every((cond) => {
      const value = cond.value;
      return !Array.isArray(value) && (value === null || ["string", "number", "boolean"].includes(typeof value));
    });

  return {
    conditions,
    hasNestedFields,
    serverCompatible,
    queryParams: serverCompatible ? buildQueryParams(conditions) : {},
  };
};

const toNumberOrDate = (value: any): number | null => {
  if (value === null || value === undefined) return null;
  if (typeof value === "number" && !Number.isNaN(value)) return value;
  if (value instanceof Date) return value.getTime();
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed.length === 0) return null;
    if (/^-?\d+(\.\d+)?$/.test(trimmed)) return Number(trimmed);
    const parsed = Date.parse(trimmed);
    return Number.isNaN(parsed) ? null : parsed;
  }
  return null;
};

const normalizeForEquality = (value: any): any => {
  if (value === null || value === undefined) return value;
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (value instanceof Date) return value.getTime();
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (/^(true|false)$/i.test(trimmed)) return trimmed.toLowerCase() === "true";
    if (/^-?\d+(\.\d+)?$/.test(trimmed)) return Number(trimmed);
    const parsed = Date.parse(trimmed);
    if (!Number.isNaN(parsed) && !/^-?\d+(\.\d+)?$/.test(trimmed)) return parsed;
    return trimmed.toLowerCase();
  }
  return value;
};

const valuesEqual = (left: any, right: any): boolean => {
  const normalizedLeft = normalizeForEquality(left);
  const normalizedRight = normalizeForEquality(right);
  return normalizedLeft === normalizedRight;
};

const matchValue = (actual: any, expected: any): boolean => {
  const actualValues = Array.isArray(actual) ? actual : [actual];
  const expectedValues = Array.isArray(expected) ? expected : [expected];
  return actualValues.some((left) => expectedValues.some((right) => valuesEqual(left, right)));
};

const evaluateCondition = (item: any, condition: FilterCondition): boolean => {
  const actual = getNestedValue(item, condition.field) ?? item?.[condition.field];
  const expected = condition.value;

  switch (condition.op) {
    case "eq":
      return matchValue(actual, expected);
    case "neq":
      return !matchValue(actual, expected);
    case "contains": {
      const actualText = Array.isArray(actual) ? actual.join(" ") : String(actual ?? "");
      const expectedText = Array.isArray(expected) ? expected.join(" ") : String(expected ?? "");
      return actualText.toLowerCase().includes(expectedText.toLowerCase());
    }
    case "in": {
      if (!Array.isArray(expected)) return matchValue(actual, expected);
      return expected.some((value) => matchValue(actual, value));
    }
    case "notin": {
      if (!Array.isArray(expected)) return !matchValue(actual, expected);
      return !expected.some((value) => matchValue(actual, value));
    }
    case "gt":
    case "gte":
    case "lt":
    case "lte": {
      const actualValues = Array.isArray(actual) ? actual : [actual];
      return actualValues.some((value) => {
        const left = toNumberOrDate(value);
        const right = toNumberOrDate(expected);
        if (left === null || right === null) return false;
        if (condition.op === "gt") return left > right;
        if (condition.op === "gte") return left >= right;
        if (condition.op === "lt") return left < right;
        return left <= right;
      });
    }
    default:
      return true;
  }
};

export const applyFilter = (data: any[], parsed: ParsedFilter | null): any[] => {
  if (!parsed || !parsed.conditions.length) return data;
  return (data || []).filter((item) => parsed.conditions.every((condition) => evaluateCondition(item, condition)));
};

/**
 * The rows of a data endpoint's response: the JSON array itself, or the array an
 * object holds under one of ROW_KEYS, else under its first array-valued key
 * (so `{"total": 3, "rows": [...]}` works too). Any other payload has no rows.
 */
export const ROW_KEYS = ["data", "results", "items", "records", "list"];

export const extractRows = (payload: any): any[] => {
  if (Array.isArray(payload)) return payload;
  if (payload && typeof payload === "object") {
    let foundKey = ROW_KEYS.find((key) => Array.isArray(payload[key]));
    if (!foundKey) {
      foundKey = Object.keys(payload).find((key) => Array.isArray(payload[key]));
    }
    if (foundKey) return payload[foundKey];
  }
  return [];
};

/** The backend's base URL (VITE_API_URL, without a trailing slash). */
export const backendBase = (): string =>
  String(import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "");

/** An endpoint as a URL: a path is joined to the backend's base, an http(s) URL is kept. */
export const backendUrl = (endpoint: string): string => {
  if (/^https?:\/\//i.test(endpoint)) return endpoint;
  return backendBase() + (endpoint.startsWith("/") ? endpoint : `/${endpoint}`);
};

const normalizeSearchPath = (pathname: string): string => {
  if (pathname.endsWith("/search") || pathname.endsWith("/search/")) {
    return pathname.endsWith("/") ? pathname : `${pathname}/`;
  }
  const base = pathname.endsWith("/") ? pathname : `${pathname}/`;
  return `${base}search/`;
};

const canUseSearchEndpoint = (pathname: string): boolean => {
  if (pathname.includes("/search/") || pathname.endsWith("/search")) return false;
  const segments = pathname.split("/").filter(Boolean);
  return segments.length === 1;
};

/**
 * The URL a chart fetches. For the class's CRUD list it may ask for detailed rows
 * and send a server-compatible filter as query parameters (to `/<class>/search/`
 * when `preferSearch`). A custom endpoint (`DataBinding.endpoint`, flagged
 * `custom`) is fetched as written: its route knows neither, so the filter is only
 * applied to the rows it returns.
 */
export const buildFilteredUrl = (
  endpoint: string,
  options: { detailed?: boolean; filter?: ParsedFilter | null; preferSearch?: boolean; custom?: boolean }
): string => {
  const isRelative = !/^https?:\/\//i.test(endpoint);
  const baseUrl = backendUrl(endpoint);
  if (options.custom) return baseUrl;
  let url: URL;
  try {
    url = new URL(baseUrl);
  } catch {
    return baseUrl;
  }

  if (options.detailed) {
    url.searchParams.set("detailed", "true");
  }

  const filter = options.filter;
  const shouldAttachFilters = !!filter?.serverCompatible && Object.keys(filter.queryParams).length > 0;
  if (shouldAttachFilters) {
    if (options.preferSearch && isRelative && canUseSearchEndpoint(url.pathname)) {
      url.pathname = normalizeSearchPath(url.pathname);
    }
    Object.entries(filter.queryParams).forEach(([key, value]) => {
      if (value !== "") {
        url.searchParams.set(key, value);
      }
    });
  }

  return url.toString();
};
