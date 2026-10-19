
import { useEffect, useRef, useState } from "react";
import { API_URL, apiRequest } from "./api";
import "./App.css";

type Video = {
  id: string;
  filename: string;
  status: string;
  created_at?: string;
};

type SearchResult = {
  detection_id: string;
  video_id: string;
  video_filename: string;
  timestamp_seconds: number;
  class_name: string;
  detection_confidence: number;
  similarity_score: number;
  bbox: number[];
  thumbnail_url: string;
};

type SearchResponse = {
  selected_video_count: number;
  result_count: number;
  results: SearchResult[];
};

export default function App() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [reference, setReference] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [activeResult, setActiveResult] =
    useState<SearchResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [query, setQuery] = useState("");

  const videoRef = useRef<HTMLVideoElement>(null);

  async function loadVideos() {
    const data = await apiRequest<{ videos: Video[] }>(
      "/api/videos",
    );
    setVideos(data.videos);
  }

  useEffect(() => {
    loadVideos().catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!reference) {
      setPreview("");
      return;
    }

    const url = URL.createObjectURL(reference);
    setPreview(url);

    return () => URL.revokeObjectURL(url);
  }, [reference]);

  async function uploadVideo() {
    if (!uploadFile) return;

    setBusy(true);
    setError("");
    setNotice("");

    try {
      const form = new FormData();
      form.append("file", uploadFile);

      await apiRequest("/api/videos", {
        method: "POST",
        body: form,
      });

      setUploadFile(null);
      await loadVideos();
      setNotice("Video uploaded. Select it and start indexing.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function indexVideo(id: string) {
    setBusy(true);
    setError("");
    setNotice("");

    try {
      await apiRequest(`/api/videos/${id}/index`, {
        method: "POST",
      });

      setNotice("Indexing started. Refresh the library to check status.");
      await loadVideos();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function search() {
    if (!reference || selected.length === 0) {
      setError("Choose a reference image and at least one video.");
      return;
    }

    setBusy(true);
    setError("");
    setNotice("");
    setResults([]);
    setActiveResult(null);

    try {
      const form = new FormData();
      form.append("file", reference);
      form.append("selected_video_ids", JSON.stringify(selected));

      const data = await apiRequest<SearchResponse>("/api/search", {
        method: "POST",
        body: form,
      });

      setResults(data.results);
      setNotice(
        `${data.result_count} matches found across ${data.selected_video_count} selected video(s).`,
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function playResult(result: SearchResult) {
    setActiveResult(result);

    // Wait for React to render the selected video source.
    requestAnimationFrame(() => {
      const video = videoRef.current;
      if (!video) return;

      video.currentTime = result.timestamp_seconds;
      void video.play().catch(() => {
        // Playback may require another click in some browsers.
      });
    });
  }

  const filteredResults = results.filter((result) =>
    `${result.class_name} ${result.video_filename}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );

  function formatTime(seconds: number) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${String(secs).padStart(2, "0")}`;
  }

  return (
    <main className="app">
      <header className="topbar">
        <div className="brand-mark">VS</div>
        <div>
          <h1>Vision Search</h1>
          <p>Find objects across your videos</p>
        </div>
        <button className="secondary refresh" onClick={() => {
          loadVideos().catch((e) => setError(e.message));
        }}>
          Refresh library
        </button>
      </header>

      <section className="intro">
        <span className="eyebrow">VIDEO INTELLIGENCE</span>
        <h2>Find the moment.<br />Not the whole video.</h2>
        <p>
          Upload your footage, choose the videos to search,
          and find visually similar objects using a reference photo.
        </p>
      </section>

      {error && <div className="message error">{error}</div>}
      {notice && <div className="message success">{notice}</div>}

      <section className="panel">
        <div className="section-heading">
          <div>
            <span className="step">01</span>
            <h3>Your video library</h3>
          </div>
          <span className="muted">{videos.length} videos</span>
        </div>

        <div className="upload-row">
          <input
            type="file"
            accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
            onChange={(e) =>
              setUploadFile(e.target.files?.[0] ?? null)
            }
          />
          <button
            onClick={uploadVideo}
            disabled={!uploadFile || busy}
          >
            Upload video
          </button>
        </div>

        <div className="video-list">
          {videos.map((video) => (
            <div className="video-row" key={video.id}>
              <input
                type="checkbox"
                checked={selected.includes(video.id)}
                onChange={(e) => {
                  setSelected((old) =>
                    e.target.checked
                      ? [...old, video.id]
                      : old.filter((id) => id !== video.id),
                  );
                }}
              />
              <div className="video-info">
                <strong>{video.filename}</strong>
                <span>{video.id}</span>
              </div>
              <span className={`status ${video.status}`}>
                {video.status}
              </span>
              {video.status !== "indexed" && (
                <button
                  className="secondary"
                  disabled={busy || video.status === "processing"}
                  onClick={() => indexVideo(video.id)}
                >
                  {video.status === "processing"
                    ? "Processing…"
                    : video.status === "failed"
                      ? "Retry index"
                      : "Index"}
                </button>
              )}
            </div>
          ))}
          {videos.length === 0 && (
            <p className="empty">Upload your first video to get started.</p>
          )}
        </div>

        <p className="hint">
          Only the videos you select will be searched.
          All selected videos must be indexed first.
        </p>
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <span className="step">02</span>
            <h3>Find an object</h3>
          </div>
        </div>

        <label className="dropzone">
          {preview ? (
            <img src={preview} alt="Reference object" />
          ) : (
            <div className="drop-placeholder">
              <span className="upload-icon">↑</span>
              <strong>Upload a reference photo</strong>
              <span>Choose an image containing the object</span>
            </div>
          )}
          <input
            type="file"
            accept="image/*"
            onChange={(e) =>
              setReference(e.target.files?.[0] ?? null)
            }
          />
        </label>

        <button
          className="search-button"
          disabled={busy || !reference || selected.length === 0}
          onClick={search}
        >
          {busy ? "Working…" : "Search selected videos"}
        </button>
      </section>

      <section className="panel results-panel">
        <div className="section-heading">
          <div>
            <span className="step">03</span>
            <h3>Search results</h3>
          </div>
          <span className="muted">{filteredResults.length} matches</span>
        </div>

        {activeResult && (
          <div className="player">
            <div className="player-heading">
              <strong>{activeResult.video_filename}</strong>
              <span>{formatTime(activeResult.timestamp_seconds)}</span>
            </div>
            <video
              ref={videoRef}
              controls
              playsInline
              src={`${API_URL}/api/videos/${activeResult.video_id}/file`}
            />
            <p className="hint">
              Playback starts at the matching timestamp.
            </p>
          </div>
        )}

        {results.length > 0 && (
          <input
            className="filter-input"
            placeholder="Filter results by object or filename..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        )}

        <div className="results-grid">
          {filteredResults.map((result) => (
            <button
              className="result-card"
              key={result.detection_id}
              onClick={() => void playResult(result)}
            >
              <img
                src={`${API_URL}${result.thumbnail_url}`}
                alt={result.class_name}
                loading="lazy"
              />
              <div className="result-body">
                <div className="result-title">
                  <strong>{result.class_name}</strong>
                  <span>{formatTime(result.timestamp_seconds)}</span>
                </div>
                <p>{result.video_filename}</p>
                <div className="score-row">
                  <span>Visual similarity</span>
                  <strong>
                    {(result.similarity_score * 100).toFixed(1)}%
                  </strong>
                </div>
                <div className="score-track">
                  <div
                    style={{
                      width: `${Math.max(
                        0,
                        Math.min(100, result.similarity_score * 100),
                      )}%`,
                    }}
                  />
                </div>
                <span className="open-result">Play this moment ↗</span>
              </div>
            </button>
          ))}
        </div>

        {results.length === 0 && (
          <div className="empty results-empty">
            <strong>Your matches will appear here.</strong>
            <p>Choose videos, upload a reference photo, and search.</p>
          </div>
        )}
      </section>
    </main>
  );
}
