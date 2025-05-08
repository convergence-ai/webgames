import React, { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_FileUploadHard = "MP3_UPLOAD_SUCCESS_2024";
export const TASK_ID_FileUploadHard = "file-upload-hard";

const FileUploadHard = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_FileUploadHard);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isCorrectFileType, setIsCorrectFileType] = useState(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);

      const reader = new FileReader();
      reader.onload = (e) => {
        const buffer = e.target?.result as ArrayBuffer;
        if (buffer) {
          const bytes = new Uint8Array(buffer);
          // Check for ID3 tag (49 44 33 in hex)
          const isID3 =
            bytes.length >= 3 &&
            bytes[0] === 0x49 &&
            bytes[1] === 0x44 &&
            bytes[2] === 0x33;
          // Check for MP3 frame sync (first 11 bits are 1s)
          // 0xFF followed by a byte starting with 0xE or 0xF (e.g., 0xFB, 0xF3, 0xF2 for MPEG Layer 3)
          const isFrameSync =
            bytes.length >= 2 &&
            bytes[0] === 0xff &&
            (bytes[1] & 0xe0) === 0xe0;

          if (
            (file.type === "audio/mpeg" ||
              file.name.toLowerCase().endsWith(".mp3")) &&
            (isID3 || isFrameSync)
          ) {
            setIsCorrectFileType(true);
            recordSuccess();
          } else {
            setIsCorrectFileType(false);
          }
        } else {
          setIsCorrectFileType(false);
        }
      };
      reader.onerror = () => {
        setIsCorrectFileType(false);
      };
      // Read the first 4 bytes of the file for checking
      reader.readAsArrayBuffer(file.slice(0, 4));
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <div className="max-w-md w-full space-y-8 p-6 bg-white rounded-lg shadow-lg">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-4">Upload Challenge (Hard)</h2>
          <p className="text-gray-600 mb-8">
            Complete this challenge by uploading a valid MP3 audio file.
          </p>
        </div>

        <div className="space-y-4">
          <div className="flex justify-center">
            <label className="relative cursor-pointer bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded-lg transition-colors">
              <span>
                {selectedFile
                  ? isCorrectFileType
                    ? "MP3 selected!"
                    : "Incorrect file type!"
                  : "Choose an MP3 file"}
              </span>
              <input
                type="file"
                className="hidden"
                onChange={handleFileChange}
                accept="audio/mpeg"
              />
            </label>
          </div>

          {selectedFile && isCorrectFileType && (
            <div className="text-center text-green-600">
              <p>✨ Challenge completed! ✨</p>
              <p className="text-sm text-gray-600">
                Uploaded: {selectedFile.name}
              </p>
              <p>
                The secret password is:{" "}
                <span className="font-bold">{PASSWORD_FileUploadHard}</span>
              </p>
            </div>
          )}
          {selectedFile && !isCorrectFileType && (
            <div className="text-center text-red-600">
              <p>Please upload an MP3 file.</p>
              <p className="text-sm text-gray-600">
                Selected: {selectedFile.name} ({selectedFile.type})
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default FileUploadHard;
