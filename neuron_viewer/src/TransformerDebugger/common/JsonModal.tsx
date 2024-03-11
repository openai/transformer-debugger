import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button,
  useDisclosure,
} from "@nextui-org/react";
import ReactJson, { OnCopyProps } from "@microlink/react-json-view";
import { MagnifyingGlassIcon } from "@heroicons/react/24/solid";
import { ExplanatoryTooltip } from "./ExplanatoryTooltip";

type JsonModalProps = {
  jsonData: any;
  buttonLabel?: string | JSX.Element;
  collapsed?: number;
};

const copyOrDownload = (copy: OnCopyProps) => {
  const jsonAsString = JSON.stringify(copy.src, null, 2);
  if (navigator.clipboard) {
    navigator.clipboard.writeText(jsonAsString).catch((err) => {
      console.error("Error in copying text: ", err);
    });
  } else {
    const blob = new Blob([jsonAsString], {
      type: "application/json",
    });
    const href = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = href;
    link.download = "data.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(href);
  }
};

export default function JsonModal({
  jsonData,
  buttonLabel = <MagnifyingGlassIcon className="h-6 w-6 text-blue-500" />,
  collapsed = 2,
}: JsonModalProps) {
  const { isOpen, onOpen, onOpenChange } = useDisclosure();

  return (
    <>
      <ExplanatoryTooltip explanation="Show the JSON used to render this component.">
        <Button onPress={onOpen}>{buttonLabel}</Button>
      </ExplanatoryTooltip>
      <Modal isOpen={isOpen} onOpenChange={onOpenChange} size={"5xl"}>
        <ModalContent>
          {(onClose) => (
            <>
              <ModalHeader className="flex flex-col gap-1">JSON Data</ModalHeader>
              <ModalBody>
                <div style={{ overflow: "auto", maxHeight: "calc(100vh - 250px)" }}>
                  <ReactJson
                    src={jsonData}
                    collapsed={collapsed}
                    enableClipboard={copyOrDownload}
                    groupArraysAfterLength={5}
                  />
                </div>
              </ModalBody>
              <ModalFooter>
                <Button color="danger" variant="light" onPress={onClose}>
                  Close
                </Button>
              </ModalFooter>
            </>
          )}
        </ModalContent>
      </Modal>
    </>
  );
}
