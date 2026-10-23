import React, { CSSProperties, useEffect, useState } from "react";
import axios from "axios";
import { backendUrl } from "./bindingData";

export interface DataSourceConfig {
  name?: string;
  domain?: string;
  fields?: string[];
  label_field?: string;
  value_field?: string;
  endpoint?: string;
}

export interface DataListBlockProps {
  id: string;
  dataSources?: DataSourceConfig[];
  style?: CSSProperties;
  className?: string;
}

export const DataListBlock: React.FC<DataListBlockProps> = ({ id, dataSources, style, className }) => {
  const [listData, setListData] = useState<any[]>([]);

  useEffect(() => {
    if (!dataSources || dataSources.length === 0) return;
    const source = dataSources[0];
    const endpoint = source.endpoint || (source.domain ? `/${source.domain.toLowerCase()}/` : "");
    if (!endpoint) return;

    const url = backendUrl(endpoint);
    axios
      .get(url)
      .then((res) => {
        const data = Array.isArray(res.data) ? res.data : res.data?.results || [];
        setListData(data);
      })
      .catch(() => setListData([]));
  }, [JSON.stringify(dataSources || [])]);

  return (
    <div id={id} style={style} className={className}>
      <ul>
        {listData.map((item, index) => (
          <li key={index}>{JSON.stringify(item)}</li>
        ))}
      </ul>
    </div>
  );
};
