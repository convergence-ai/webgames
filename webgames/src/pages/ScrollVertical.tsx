export const PASSWORD_ScrollVertical = "SCROLLMASTER2024";

const ScrollVertical = () => {
  // Generate content boxes
  const boxes = Array(50)
    .fill(null)
    .map((_, index) => (
      <div
        key={index}
        style={{
          height: "200px",
          margin: "20px",
          backgroundColor: `hsl(${(index * 7) % 360}, 70%, 80%)`,
          borderRadius: "8px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "24px",
          color: "#444",
        }}
      >
        Keep scrolling! {50 - index} boxes to go...
      </div>
    ));

  return (
    <div style={{ padding: "20px" }}>
      <h1>Scroll to the Bottom</h1>
      <p>Keep scrolling down to reveal the secret password!</p>

      {boxes}

      <div
        style={{
          padding: "20px",
          backgroundColor: "#4CAF50",
          color: "white",
          borderRadius: "8px",
          textAlign: "center",
          marginTop: "20px",
        }}
      >
        Congratulations! You've reached the bottom!
        <div
          style={{ marginTop: "10px", fontSize: "24px", fontWeight: "bold" }}
        >
          Password: {PASSWORD_ScrollVertical}
        </div>
      </div>
    </div>
  );
};

export default ScrollVertical;
