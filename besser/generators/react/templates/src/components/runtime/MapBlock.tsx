import React, { CSSProperties, useEffect, useState } from "react";
import axios from "axios";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Fix Leaflet's default marker icon — webpack / vite asset handling breaks the
// built-in icon URL resolution, so we point it at the CDN copies explicitly.
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

export interface MapBlockProps {
  id: string;
  title?: string;
  /** Static map config: center, zoom, and geo-field names */
  mapConfig?: {
    centerLatitude?: number;
    centerLongitude?: number;
    zoom?: number;
    latitudeField?: string;
    longitudeField?: string;
    markerLabelField?: string;
  };
  /** DataBinding produced by the BESSER generator (endpoint + entity) */
  dataBinding?: Record<string, any>;
  styles?: CSSProperties;
}

/**
 * MapBlock — renders an OpenStreetMap / Leaflet map.
 *
 * When `dataBinding.endpoint` is provided the component fetches rows from the
 * backend REST API and places a marker for each row whose latitude/longitude
 * fields are populated.  Without a binding it shows a plain tile map centred
 * on the configured coordinates.
 */
export const MapBlock: React.FC<MapBlockProps> = ({
  id,
  title,
  mapConfig,
  dataBinding,
  styles,
}) => {
  const center: [number, number] = [
    mapConfig?.centerLatitude ?? 0,
    mapConfig?.centerLongitude ?? 0,
  ];
  const zoom = mapConfig?.zoom ?? 10;
  const latField = mapConfig?.latitudeField;
  const lngField = mapConfig?.longitudeField;
  const labelField = mapConfig?.markerLabelField;

  const [markers, setMarkers] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const endpoint = dataBinding?.endpoint;
    if (!endpoint || !latField || !lngField) {
      setMarkers([]);
      return;
    }

    setLoading(true);
    setError(null);

    const backendBase = import.meta.env.VITE_API_URL || "http://localhost:8000";
    const url = endpoint.startsWith("/") ? backendBase + endpoint : endpoint;

    axios
      .get(url)
      .then((res) => {
        let rows: any[] = [];
        if (Array.isArray(res.data)) {
          rows = res.data;
        } else if (res.data && typeof res.data === "object") {
          const commonKeys = ["data", "results", "items", "records", "list"];
          const foundKey =
            commonKeys.find((k) => Array.isArray((res.data as any)[k])) ||
            Object.keys(res.data).find((k) => Array.isArray((res.data as any)[k]));
          if (foundKey) rows = (res.data as any)[foundKey];
        }
        setMarkers(rows);
      })
      .catch((err) => {
        console.error("[MapBlock] Error loading marker data:", err);
        setError("Error loading map data");
        setMarkers([]);
      })
      .finally(() => setLoading(false));
  }, [dataBinding?.endpoint, latField, lngField]);

  if (loading) return <div id={id} style={styles}>Loading map data…</div>;
  if (error)   return <div id={id} style={styles}>{error}</div>;

  const containerStyle: CSSProperties = {
    width: "100%",
    height: "450px",
    ...styles,
  };

  return (
    <div id={id} style={{ width: "100%" }}>
      {title && (
        <h3 style={{ margin: "0 0 8px", fontSize: "1rem", fontWeight: 600 }}>
          {title}
        </h3>
      )}
      <MapContainer
        center={center}
        zoom={zoom}
        style={containerStyle}
        scrollWheelZoom={false}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {markers.map((row, idx) => {
          const lat = latField ? parseFloat(row[latField]) : NaN;
          const lng = lngField ? parseFloat(row[lngField]) : NaN;
          if (isNaN(lat) || isNaN(lng)) return null;
          const label = labelField ? row[labelField] : undefined;
          return (
            <Marker key={idx} position={[lat, lng]}>
              {label !== undefined && (
                <Popup>{String(label)}</Popup>
              )}
            </Marker>
          );
        })}
      </MapContainer>
    </div>
  );
};
