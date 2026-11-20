import React, { CSSProperties, useEffect, useState } from "react";
import { apiRequest } from "../../api/generated/client";
import { useDataVersion } from "./dataVersion";

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
  const dataVersion = useDataVersion();

  useEffect(() => {
    if (!dataSources || dataSources.length === 0) return;
    const source = dataSources[0];
    const endpoint = source.endpoint || (source.domain ? `/${source.domain.toLowerCase()}/` : "");
    if (!endpoint) return;

    apiRequest<any>("GET", endpoint)
      .then((body) => {
        const data = Array.isArray(body) ? body : body?.results || [];
        setListData(data);
      })
      .catch(() => setListData([]));
  }, [JSON.stringify(dataSources || []), dataVersion]);

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
