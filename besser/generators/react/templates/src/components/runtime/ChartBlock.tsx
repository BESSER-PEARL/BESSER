import React, { CSSProperties, useEffect, useMemo, useState } from "react";
import axios from "axios";
import { BarChartComponent } from "../charts/BarChartComponent";
import { LineChartComponent } from "../charts/LineChartComponent";
import { PieChartComponent } from "../charts/PieChartComponent";
import { RadarChartComponent } from "../charts/RadarChartComponent";
import { RadialBarChartComponent } from "../charts/RadialBarChartComponent";
import { aggregate, normalizeAggregation } from "./aggregate";
import {
  applyFilter,
  buildFilteredUrl,
  extractRows,
  getNestedValue,
  isNestedField,
  parseFilterExpression,
} from "./bindingData";

export interface ChartSeries {
  name?: string;
  label?: string;
  color?: string;
  dataSource?: string;
  endpoint?: string;
  // The endpoint is a DataBinding.endpoint, not the class's CRUD list: fetched as written
  customEndpoint?: boolean;
  labelField?: string;
  dataField?: string;
  "label-field"?: string;
  "data-field"?: string;
  "data-source"?: string;
  filter?: string;
  // Groups the records by labelField and reduces each group (count, sum, avg...)
  aggregation?: string;
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
  className?: string;
}


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

export const ChartBlock: React.FC<ChartBlockProps> = ({
  id,
  chartType,
  title,
  color,
  chart,
  series,
  dataBinding,
  styles,
  className,
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
      // A bound series with no value field counts the records per label
      aggregation: normalizeAggregation(s?.aggregation) ?? (
        (s?.endpoint || s?.dataSource) && !(s?.dataField || s?.["data-field"] || dataBinding?.data_field)
          ? "count"
          : undefined
      ),
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

    const fetchPromises = seriesWithEndpoints.map((s) => {
      const seriesName = s.name || s.label || "Series";
      const endpoint = s.endpoint || `/${s.dataSource || (s as any)["data-source"]}/`;
      const parsedFilter = parseFilterExpression(s.filter);
      const needsDetailed =
        isNestedField(s.labelField) || isNestedField(s.dataField) || Boolean(parsedFilter?.hasNestedFields);
      const url = buildFilteredUrl(endpoint, {
        detailed: needsDetailed,
        filter: parsedFilter,
        preferSearch: !!parsedFilter?.serverCompatible && !needsDetailed,
        custom: Boolean(s.customEndpoint),
      });
      return axios
        .get(url)
        .then((res) => {
          const data = applyFilter(extractRows(res.data), parsedFilter);
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
    const filterExpression = dataBinding?.filter || dataBinding?.filter_expression;
    const parsedFilter = parseFilterExpression(filterExpression);
    const needsDetailed =
      isNestedField(dataBinding?.label_field) ||
      isNestedField(dataBinding?.data_field) ||
      Boolean(parsedFilter?.hasNestedFields);
    const url = buildFilteredUrl(endpoint, {
      detailed: needsDetailed,
      filter: parsedFilter,
      preferSearch: !!parsedFilter?.serverCompatible && !needsDetailed,
      custom: Boolean(dataBinding?.custom_endpoint),
    });

    axios
      .get(url)
      .then((res) => {
        const data = applyFilter(extractRows(res.data), parsedFilter);
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
    dataBinding?.custom_endpoint,
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
      const aggregation = normalizeAggregation(s.aggregation);
      if (aggregation) {
        const groups: Record<string, any[]> = {};
        filteredData.forEach((item: any) => {
          const key = String(getNestedValue(item, s.labelField || "name") ?? "");
          if (!groups[key]) groups[key] = [];
          groups[key].push(item);
        });
        Object.entries(groups).forEach(([key, rows]) => {
          if (!combined[key]) combined[key] = { name: key };
          combined[key][s.name || "Series"] = aggregate(rows, aggregation, s.dataField);
        });
        return;
      }
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

  if (loading) return <div id={id} className={className}>Loading data...</div>;
  if (error) return <div id={id} className={className}>{error}</div>;

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
        className={className}
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
        className={className}
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
        className={className}
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
        className={className}
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
        className={className}
      />
    );
  }

  return null;
};
