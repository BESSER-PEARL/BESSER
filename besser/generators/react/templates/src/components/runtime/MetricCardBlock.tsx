import React, { CSSProperties, useEffect, useState } from "react";
import axios from "axios";
import { MetricCardComponent } from "../charts/MetricCardComponent";
import { aggregate, normalizeAggregation } from "./aggregate";

export interface MetricCardBlockProps {
  id: string;
  metric?: Record<string, any>;
  dataBinding?: Record<string, any>;
  styles?: CSSProperties;
  className?: string;
}

// The card's value over all records: its aggregation over the bound field,
// else the field's sum, else the number of records.
const metricValue = (data: any[], dataField?: string, aggregation?: string): number => {
  const resolved = normalizeAggregation(aggregation) ?? (dataField ? "sum" : "count");
  return aggregate(data || [], resolved, dataField);
};

export const MetricCardBlock: React.FC<MetricCardBlockProps> = ({
  id,
  metric,
  dataBinding,
  styles,
  className,
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
        setValue(metricValue(data, dataBinding?.data_field, dataBinding?.aggregation));
      })
      .catch(() => {
        setValue(metric?.value ?? 0);
      })
      .finally(() => setLoading(false));
  }, [dataBinding?.endpoint, dataBinding?.data_field, dataBinding?.aggregation, metric?.value]);

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
      className={className}
    />
  );
};
