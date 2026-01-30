import React, { CSSProperties, useEffect, useState } from "react";
import axios from "axios";
import { MetricCardComponent } from "../charts/MetricCardComponent";

export interface MetricCardBlockProps {
  id: string;
  metric?: Record<string, any>;
  dataBinding?: Record<string, any>;
  styles?: CSSProperties;
}

const getLastValue = (data: any[], dataField?: string): number => {
  if (!data || data.length === 0) return 0;
  const lastItem = data[data.length - 1];
  if (dataField && lastItem[dataField] !== undefined) {
    return Number(lastItem[dataField]) || 0;
  }
  const commonFields = ["value", "count", "amount", "total", "sum"];
  for (const field of commonFields) {
    if (lastItem[field] !== undefined) {
      return Number(lastItem[field]) || 0;
    }
  }
  return 0;
};

export const MetricCardBlock: React.FC<MetricCardBlockProps> = ({
  id,
  metric,
  dataBinding,
  styles,
}) => {
  const [value, setValue] = useState<number>(metric?.value ?? 0);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const endpoint = dataBinding?.endpoint;
    if (!endpoint) return;
    setLoading(true);
    const backendBase = import.meta.env.VITE_API_URL || "http://localhost:8000";
    const url = endpoint.startsWith("/") ? backendBase + endpoint : endpoint;

    axios
      .get(url)
      .then((res) => {
        let data: any[] = [];
        if (Array.isArray(res.data)) {
          data = res.data;
        } else if (res.data && typeof res.data === "object") {
          const commonArrayKeys = ["data", "results", "items", "records", "list"];
          let foundKey = commonArrayKeys.find((key) => Array.isArray((res.data as any)[key]));
          if (!foundKey) {
            foundKey = Object.keys(res.data).find((key) => Array.isArray((res.data as any)[key]));
          }
          if (foundKey) {
            data = (res.data as any)[foundKey];
          }
        }
        setValue(getLastValue(data, dataBinding?.data_field));
      })
      .catch(() => {
        setValue(metric?.value ?? 0);
      })
      .finally(() => setLoading(false));
  }, [dataBinding?.endpoint, dataBinding?.data_field, metric?.value]);

  return (
    <MetricCardComponent
      id={id}
      metric-title={metric?.metricTitle}
      format={metric?.format}
      value-color={metric?.valueColor}
      value-size={metric?.valueSize}
      show-trend={metric?.showTrend}
      positive-color={metric?.positiveColor}
      negative-color={metric?.negativeColor}
      value={loading ? metric?.value ?? 0 : value}
      trend={metric?.trend ?? 0}
      data_binding={dataBinding}
      styles={styles}
    />
  );
};
