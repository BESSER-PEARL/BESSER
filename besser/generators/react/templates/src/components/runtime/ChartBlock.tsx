import React, { CSSProperties, useEffect, useMemo, useState } from "react";
import axios from "axios";
import { BarChartComponent } from "../charts/BarChartComponent";
import { LineChartComponent } from "../charts/LineChartComponent";
import { PieChartComponent } from "../charts/PieChartComponent";
import { RadarChartComponent } from "../charts/RadarChartComponent";
import { RadialBarChartComponent } from "../charts/RadialBarChartComponent";

export interface ChartSeries {
  name?: string;
  label?: string;
  color?: string;
  dataSource?: string;
  endpoint?: string;
  labelField?: string;
  dataField?: string;
  "label-field"?: string;
  "data-field"?: string;
  "data-source"?: string;
  filter?: string;
  data?: any[];
}

export interface ChartBlockProps {
  id: string;
  chartType: "bar-chart" | "line-chart" | "pie-chart" | "radar-chart" | "radial-bar-chart";
  title?: string;
  color?: string;
  chart?: Record<string, any>;
  series?: ChartSeries[];
  dataBinding?: Record<string, any>;
  styles?: CSSProperties;
}

const isNestedField = (field?: string): boolean => !!field && field.includes(".");

const getNestedValue = (obj: any, path: string): any => {
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

const parseSeries = (rawSeries: any): ChartSeries[] => {
  if (!rawSeries) return [];
  if (Array.isArray(rawSeries)) return rawSeries;
  if (typeof rawSeries === "string") {
    try {
      const parsed = JSON.parse(rawSeries);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }
  return [];
};

type FilterOperator = "eq" | "neq" | "gt" | "gte" | "lt" | "lte" | "contains" | "in" | "notin";

interface FilterCondition {
  field: string;
  op: FilterOperator;
  value: any;
}

interface ParsedFilter {
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

const parseFilterExpression = (filter: unknown): ParsedFilter | null => {
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

const applyFilter = (data: any[], parsed: ParsedFilter | null): any[] => {
  if (!parsed || !parsed.conditions.length) return data;
  return (data || []).filter((item) => parsed.conditions.every((condition) => evaluateCondition(item, condition)));
};

const extractArrayFromResponse = (payload: any): any[] => {
  if (Array.isArray(payload)) return payload;
  if (payload && typeof payload === "object") {
    const commonArrayKeys = ["data", "results", "items", "records", "list"];
    let foundKey = commonArrayKeys.find((key) => Array.isArray(payload[key]));
    if (!foundKey) {
      foundKey = Object.keys(payload).find((key) => Array.isArray(payload[key]));
    }
    if (foundKey) return payload[foundKey];
  }
  return [];
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

const buildFilteredUrl = (
  endpoint: string,
  backendBase: string,
  options: { detailed?: boolean; filter?: ParsedFilter | null; preferSearch?: boolean }
): string => {
  const isAbsolute = /^https?:\/\//i.test(endpoint);
  const normalizedEndpoint = endpoint.startsWith("/") || isAbsolute ? endpoint : `/${endpoint}`;
  const isRelative = !isAbsolute;
  const baseUrl = isRelative ? backendBase.replace(/\/$/, "") + normalizedEndpoint : endpoint;
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

export const ChartBlock: React.FC<ChartBlockProps> = ({
  id,
  chartType,
  title,
  color,
  chart,
  series,
  dataBinding,
  styles,
}) => {
  const [chartData, setChartData] = useState<any[]>([]);
  const [seriesData, setSeriesData] = useState<Record<string, any[]>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const normalizedSeries = useMemo(() => {
    const parsed = parseSeries(series);
    return parsed.map((s, index) => ({
      ...s,
      name: s?.name || s?.label || `Series ${index + 1}`,
      labelField: s?.labelField || s?.["label-field"] || dataBinding?.label_field || "name",
      dataField: s?.dataField || s?.["data-field"] || dataBinding?.data_field || "value",
      filter: s?.filter || (s as any)?.["filter"],
      fetchedData: (s as any).fetchedData,
    }));
  }, [series, dataBinding]);

  useEffect(() => {
    if (!["bar-chart", "line-chart", "pie-chart", "radial-bar-chart", "radar-chart"].includes(chartType)) {
      return;
    }
    if (!normalizedSeries.length) return;

    const seriesWithEndpoints = normalizedSeries.filter((s) => s.endpoint || s.dataSource || (s as any)["data-source"]);
    if (!seriesWithEndpoints.length) return;

    setLoading(true);
    setError(null);
    const backendBase = import.meta.env.VITE_API_URL || "http://localhost:8000";

    const fetchPromises = seriesWithEndpoints.map((s) => {
      const seriesName = s.name || s.label || "Series";
      const endpoint = s.endpoint || `/${s.dataSource || (s as any)["data-source"]}/`;
      const parsedFilter = parseFilterExpression(s.filter);
      const needsDetailed =
        isNestedField(s.labelField) || isNestedField(s.dataField) || Boolean(parsedFilter?.hasNestedFields);
      const url = buildFilteredUrl(endpoint, backendBase, {
        detailed: needsDetailed,
        filter: parsedFilter,
        preferSearch: !!parsedFilter?.serverCompatible && !needsDetailed,
      });
      return axios
        .get(url)
        .then((res) => {
          const data = applyFilter(extractArrayFromResponse(res.data), parsedFilter);
          return { seriesName, data };
        })
        .catch(() => ({ seriesName, data: [] }));
    });

    Promise.all(fetchPromises)
      .then((results) => {
        const dataMap: Record<string, any[]> = {};
        results.forEach((result) => {
          dataMap[result.seriesName] = result.data;
        });
        setSeriesData(dataMap);
      })
      .finally(() => setLoading(false));
  }, [chartType, normalizedSeries]);

  useEffect(() => {
    const endpoint = dataBinding?.endpoint;
    if (!endpoint) return;

    setLoading(true);
    setError(null);
    const backendBase = import.meta.env.VITE_API_URL || "http://localhost:8000";
    const filterExpression = dataBinding?.filter || dataBinding?.filter_expression;
    const parsedFilter = parseFilterExpression(filterExpression);
    const needsDetailed =
      isNestedField(dataBinding?.label_field) ||
      isNestedField(dataBinding?.data_field) ||
      Boolean(parsedFilter?.hasNestedFields);
    const url = buildFilteredUrl(endpoint, backendBase, {
      detailed: needsDetailed,
      filter: parsedFilter,
      preferSearch: !!parsedFilter?.serverCompatible && !needsDetailed,
    });

    axios
      .get(url)
      .then((res) => {
        const data = applyFilter(extractArrayFromResponse(res.data), parsedFilter);
        setChartData(data);
      })
      .catch((err) => {
        console.error("[ChartBlock] Error loading data:", err);
        setError("Error loading data");
        setChartData([]);
      })
      .finally(() => setLoading(false));
  }, [
    dataBinding?.endpoint,
    dataBinding?.filter,
    dataBinding?.filter_expression,
    dataBinding?.label_field,
    dataBinding?.data_field,
  ]);

  const prepareChartDataFromSeries = (seriesConfig: ChartSeries[], seriesDataMap: Record<string, any[]>) => {
    if (!seriesConfig.length) return [];
    const combined: Record<string, any> = {};
    seriesConfig.forEach((s) => {
      const sourceData = seriesDataMap[s.name || ""] ?? (s as any).fetchedData ?? s.data ?? [];
      if (!Array.isArray(sourceData)) return;
      const parsedFilter = parseFilterExpression(s.filter);
      const filteredData = applyFilter(sourceData, parsedFilter);
      filteredData.forEach((item: any) => {
        const label = getNestedValue(item, s.labelField || "name") ?? item?.[s.labelField || "name"] ?? "";
        const value = Number(getNestedValue(item, s.dataField || "value") ?? item?.[s.dataField || "value"] ?? 0);
        const key = String(label ?? "");
        if (!combined[key]) combined[key] = { name: key };
        combined[key][s.name || "Series"] = value;
      });
    });
    return Object.values(combined);
  };

  const hasSeries = normalizedSeries.length > 0;
  const seriesChartData = hasSeries ? prepareChartDataFromSeries(normalizedSeries, seriesData) : [];
  const defaultLabelField = dataBinding?.label_field || "name";
  const defaultDataField = dataBinding?.data_field || "value";
  const resolvedLabelField = hasSeries ? "name" : defaultLabelField;
  const resolvedDataField = hasSeries
    ? (normalizedSeries.length === 1 ? normalizedSeries[0]?.name || defaultDataField : defaultDataField)
    : defaultDataField;
  const finalChartData = hasSeries ? (seriesChartData.length > 0 ? seriesChartData : chartData) : chartData;

  if (loading) return <div id={id}>Loading data...</div>;
  if (error) return <div id={id}>{error}</div>;

  if (chartType === "bar-chart") {
    return (
      <BarChartComponent
        id={id}
        title={title}
        color={color}
        data={finalChartData}
        series={normalizedSeries}
        labelField={resolvedLabelField}
        dataField={resolvedDataField}
        options={chart || {}}
        styles={styles}
      />
    );
  }

  if (chartType === "line-chart") {
    return (
      <LineChartComponent
        id={id}
        title={title}
        color={color}
        data={finalChartData}
        series={normalizedSeries}
        labelField={resolvedLabelField}
        dataField={resolvedDataField}
        options={chart || {}}
        styles={styles}
      />
    );
  }

  if (chartType === "pie-chart") {
    return (
      <PieChartComponent
        id={id}
        title={title}
        data={finalChartData}
        series={normalizedSeries}
        labelField={resolvedLabelField}
        dataField={resolvedDataField}
        options={chart || {}}
        styles={styles}
      />
    );
  }

  if (chartType === "radar-chart") {
    return (
      <RadarChartComponent
        id={id}
        title={title}
        color={color}
        data={finalChartData}
        series={normalizedSeries}
        labelField={resolvedLabelField}
        dataField={resolvedDataField}
        options={chart || {}}
        styles={styles}
      />
    );
  }

  if (chartType === "radial-bar-chart") {
    return (
      <RadialBarChartComponent
        id={id}
        title={title}
        color={color}
        data={finalChartData}
        series={normalizedSeries}
        labelField={resolvedLabelField}
        dataField={resolvedDataField}
        options={chart || {}}
        styles={styles}
      />
    );
  }

  return null;
};
