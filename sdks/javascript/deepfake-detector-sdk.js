/**
 * Antigravity Deepfake Shield - JavaScript Client SDK
 */
class DeepfakeDetectorSDK {
  constructor(endpointUrl = "http://127.0.0.1:8000") {
    this.baseUrl = endpointUrl.replace(/\/$/, "");
  }

  /**
   * Uploads file for tiered deepfake detection analysis.
   */
  async detectFile(file, isVideo = false, isSynthetic = false) {
    const url = `${this.baseUrl}/detect/upload`;
    const formData = new FormData();
    formData.append("file", file);
    formData.append("is_video", String(isVideo));
    formData.append("is_synthetic", String(isSynthetic));

    const response = await fetch(url, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Detection request failed with status: ${response.status}`);
    }

    return await response.json();
  }

  /**
   * Submits a list of media URLs for processing.
   */
  async submitBatch(mediaUrls, priority = "medium", webhookUrl = null) {
    const url = `${this.baseUrl}/detect/batch`;
    const payload = {
      media_urls: mediaUrls,
      priority: priority,
      webhook_url: webhookUrl,
    };

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Batch request failed with status: ${response.status}`);
    }

    return await response.json();
  }

  /**
   * Uploads an image, signs/embeds C2PA credentials, and returns the response blob.
   */
  async embedC2PA(file, creator = "JS SDK Author", editSummary = "Edited via JS SDK") {
    const url = `${this.baseUrl}/c2pa/embed`;
    const formData = new FormData();
    formData.append("file", file);
    formData.append("creator", creator);
    formData.append("edit_summary", editSummary);

    const response = await fetch(url, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`C2PA embed failed with status: ${response.status}`);
    }

    return await response.blob();
  }

  /**
   * Extracts and verifies C2PA credentials and blockchain anchors of a file.
   */
  async verifyC2PA(file) {
    const url = `${this.baseUrl}/c2pa/verify`;
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(url, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`C2PA verification failed with status: ${response.status}`);
    }

    return await response.json();
  }

  /**
   * Queries model zoo registered detectors and history.
   */
  async getModelZoo() {
    const url = `${this.baseUrl}/zoo/models`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Model zoo fetch failed with status: ${response.status}`);
    }

    return await response.json();
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = DeepfakeDetectorSDK;
} else {
  window.DeepfakeDetectorSDK = DeepfakeDetectorSDK;
}
