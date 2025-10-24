export function transformStyles(styles: any[]) {
  return styles.map((s) => {
    const size = s.styling.size || {};
    const position = s.styling.position || {};
    const color = s.styling.color || {};
    const styleObj: Record<string, any> = {
      width: size.width,
      height: size.height,
      padding: size.padding,
      margin: size.margin,
      fontSize: size.font_size,
      backgroundColor: color.background_color,
      color: color.text_color,
      borderColor: color.border_color,
      position: position.p_type,
      top: position.top,
      left: position.left,
      right: position.right,
      bottom: position.bottom,
      zIndex: position.z_index,
    };
    // Add textAlign if alignment is present
    if (position.alignment) {
      styleObj.textAlign = position.alignment;
    }
    return {
      selectors: [`#${s.id}`],
      style: styleObj,
    };
  });
}