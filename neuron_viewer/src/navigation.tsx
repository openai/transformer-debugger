import React, { useEffect, useCallback } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { Node } from "./types";

type NavigationProps = {
  activeNode: Node;
};

const Navigation: React.FC<NavigationProps> = ({ activeNode }) => {
  const location = useLocation();

  const getPath = useCallback(
    (nodeIndexOffset: number, layerIndexOffset: number) => {
      const currentPath = location.pathname;
      const currentPathWithoutNode = currentPath.substring(0, currentPath.lastIndexOf("/"));
      const currentPathWithoutLayer = currentPathWithoutNode.substring(
        0,
        currentPathWithoutNode.lastIndexOf("/")
      );
      const newLayerIndex = activeNode.layerIndex + layerIndexOffset;
      const newNodeIndex = activeNode.nodeIndex + nodeIndexOffset;
      // queryString is the part of the URL after the ?, which typically contains the prompt of interest
      const queryString = location.search;
      return `${currentPathWithoutLayer}/${newLayerIndex}/${newNodeIndex}${queryString}`;
    },
    [activeNode, location]
  );

  const navigate = useNavigate();
  const nextNodePath = getPath(1, 0);
  const previousNodePath = getPath(-1, 0);
  const nextLayerPath = getPath(0, 1);
  const previousLayerPath = getPath(0, -1);

  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }

      switch (event.key) {
        case "ArrowLeft":
          event.preventDefault();
          navigate(previousNodePath);
          break;
        case "ArrowRight":
          event.preventDefault();
          navigate(nextNodePath);
          break;
        case "ArrowUp":
          event.preventDefault();
          navigate(nextLayerPath);
          break;
        case "ArrowDown":
          event.preventDefault();
          navigate(previousLayerPath);
          break;
      }
    };
    window.addEventListener("keydown", handleKeyPress);
    return () => {
      window.removeEventListener("keydown", handleKeyPress);
    };
  }, [previousNodePath, nextNodePath, nextLayerPath, previousLayerPath, navigate]);

  return (
    <div style={{ position: "absolute", top: 0, left: 15 }}>
      <Link className="inline-block mr-4 pt-4 text-blue-500 underline" to="/">
        Home
      </Link>
      <Link className="inline-block mr-4 pt-4 text-blue-500 underline" to={previousNodePath}>
        Previous Node (&#8592;)
      </Link>
      <Link className="inline-block mr-4 pt-4 text-blue-500 underline" to={nextNodePath}>
        Next Node (&#8594;)
      </Link>
      <Link className="inline-block mr-4 pt-4 text-blue-500 underline" to={previousLayerPath}>
        Previous Layer (&#8595;)
      </Link>
      <Link className="inline-block mr-4 pt-4 text-blue-500 underline" to={nextLayerPath}>
        Next Layer (&#8593;)
      </Link>
      <h3 className="flex flex-row inline-block mr-4 pt-4">
        {activeNode.layerIndex}:{activeNode.nodeIndex}
      </h3>
    </div>
  );
};

export default Navigation;
