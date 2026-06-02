import { InlineMarkdown } from "./InlineMarkdown";

/** Renders a bullet as markdown text or as MathML + optional caption. */
export function SlideBullet({ point }) {
  if (typeof point === "string") {
    return <InlineMarkdown text={point} />;
  }

  if (point && typeof point === "object" && point.mathml) {
    return (
      <>
        <span
          className="inline-block align-middle text-white"
          dangerouslySetInnerHTML={{ __html: point.mathml }}
        />
        {point.text && (
          <span className="ml-1">
            <InlineMarkdown text={point.text} />
          </span>
        )}
      </>
    );
  }

  return String(point ?? "");
}
