import React from "react";
import componentsData from "../data/ui_components.json";
import stylesData from "../data/ui_styles.json";
import { Renderer } from "../components/Renderer";
import { transformStyles } from "../utils/transformStyles";

const Home: React.FC = () => {
  const page = componentsData.pages.find((p) => p.id === "page_main");
  if (!page) return <div>Error: page not found.</div>;

  const components = page.components || [];
  const transformedStyles = transformStyles(stylesData.styles);

  return (
    <div style={{ width: "100%", minHeight: "100vh", background: "#f8f8f8", padding: "20px" }}>
      {components.map((component) => (
        <Renderer key={component.id} component={component} styles={transformedStyles} />
      ))}
    </div>
  );
};

export default Home;