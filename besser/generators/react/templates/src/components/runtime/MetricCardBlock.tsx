import React, { CSSProperties, useEffect, useState } from "react";
import axios from "axios";
import { MetricCardComponent } from "../charts/MetricCardComponent";
import { aggregate, normalizeAggregation } from "./aggregate";
import { applyFilter, backendUrl, extractRows, parseFilterExpression } from "./bindingData";

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
    // The filter applies to the rows the endpoint returns, as on a chart.
    const parsedFilter = parseFilterExpression(dataBinding?.filter_expression || dataBinding?.filter);

    axios
      .get(backendUrl(endpoint))
      .then((res) => {
        const data = applyFilter(extractRows(res.data), parsedFilter);
        setValue(metricValue(data, dataBinding?.data_field, dataBinding?.aggregation));
      })
      .catch(() => {
        setValue(metric?.value ?? 0);
      })
      .finally(() => setLoading(false));
  }, [
    dataBinding?.endpoint,
    dataBinding?.filter,
    dataBinding?.filter_expression,
    dataBinding?.data_field,
    dataBinding?.aggregation,
    metric?.value,
  ]);

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
