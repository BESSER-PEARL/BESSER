import React from "react";
import { Renderer } from "./Renderer";
import { applyStyle } from "../utils/applyStyle";


interface WrapperProps {
  id: string;
  components?: any[];
  styles: any[];
}

export const Wrapper: React.FC<WrapperProps> = ({ id, components, styles }) => {
  const style = applyStyle(`#${id}`, styles);
  return (
    <div id={id} style={style}>
      {components?.map((child) => (
        <Renderer key={child.id} component={child} styles={styles} />
      ))}
    </div>
  );
};