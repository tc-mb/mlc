package com.modelbest.minicpm

import com.modelbest.mlcllm.ChatModule
import android.annotation.SuppressLint
import android.app.Application
import android.app.DownloadManager
import android.content.BroadcastReceiver
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.net.Uri
import android.os.Environment
import android.widget.Toast
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.toMutableStateList
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import kotlinx.coroutines.launch
import java.io.File
import java.util.Timer
import java.util.TimerTask
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.Future
import kotlin.concurrent.thread


class AppViewModel(application: Application) : AndroidViewModel(application) {
    val modelList = emptyList<ModelState>().toMutableStateList()
    val chatState = ChatState()
    val modelSampleList = emptyList<ModelRecord>().toMutableStateList()
    private var showAlert = mutableStateOf(false)
    private var alertMessage = mutableStateOf("")
    private var appConfig = AppConfig(
        emptyList(),
        emptyList<ModelRecord>().toMutableList()
    )
    private lateinit var downloadManager: DownloadManager

    private val application = getApplication<Application>()
    private val appDirFile = application.getExternalFilesDir("")
    private val appCacheFile = appDirFile // application.cacheDir
    private val gson = Gson()
    private val localIdSet = emptySet<String>().toMutableSet()

    companion object {
        const val AppConfigFilename = "app-config.json"
        const val ModelConfigFilename = "mlc-chat-config.json"
        const val ParamsConfigFilename = "ndarray-cache.json"
        const val ModelUrlSuffix = ""
    }

    init {
        downloadManager = application.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
        loadAppConfig()
    }

    fun isShowingAlert(): Boolean {
        return showAlert.value
    }

    fun errorMessage(): String {
        return alertMessage.value
    }

    fun dismissAlert() {
        require(showAlert.value)
        showAlert.value = false
    }

    fun copyError() {
        require(showAlert.value)
        val clipboard =
            application.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(ClipData.newPlainText("MLCChat", errorMessage()))
    }

    private fun issueAlert(error: String) {
        showAlert.value = true
        alertMessage.value = error
    }

    fun requestDeleteModel(localId: String) {
        deleteModel(localId)
        issueAlert("Model: $localId has been deleted")
    }

    private fun extract_from_asset(srcPath: String, dstPath: String) {
        val srcPath = srcPath.lowercase()
        val assetManager = application.assets
        File(dstPath).mkdirs()
        val files = assetManager.list(srcPath)
        val ts = mutableListOf<Thread>()
        for (file in files!!) {
            if (File(dstPath + '/' + file).exists()) continue
            ts.add(thread(start=true){
                //  https://stackoverflow.com/questions/43894100/best-practice-for-converting-java-code-used-for-copying-assets-files-to-cache-fo
                val inputStream = assetManager.open(srcPath + '/' + file).use { input ->
                    val bufferedOutputStream = File(dstPath + '/' + file).outputStream().buffered().use { output ->
                        input.copyTo(output, 10240)
                    }
                }
            })
        }
        for (t in ts) {
            t.join()
        }
    }

    private fun loadAppConfig() {
        val appConfigFile = File(appDirFile, AppConfigFilename)
        val jsonString: String = if (!appConfigFile.exists()) {
            application.assets.open(AppConfigFilename).bufferedReader().use { it.readText() }
        } else {
            appConfigFile.readText()
        }
        appConfig = gson.fromJson(jsonString, AppConfig::class.java)
        modelList.clear()
        localIdSet.clear()
        modelSampleList.clear()
        for (modelRecord in appConfig.modelList) {
            extract_from_asset(modelRecord.localId, appCacheFile?.absolutePath+'/'+modelRecord.localId)
//            download_from_web(modelRecord.modelUrl, appCacheFile?.absolutePath+'/'+modelRecord.localId)
            val modelDirFile = File(appCacheFile, modelRecord.localId)
            val modelConfigFile = File(modelDirFile, ModelConfigFilename)

            val modelConfigString = modelConfigFile.readText()
            val modelConfig = gson.fromJson(modelConfigString, ModelConfig::class.java)
            modelConfig.localId = modelRecord.localId
            modelConfig.modelLib = modelRecord.modelLib
            addModelConfig(modelConfig, modelRecord.modelUrl, true)
        }
    }

    private fun updateAppConfig(action: () -> Unit) {
        action()
        val jsonString = gson.toJson(appConfig)
        val appConfigFile = File(appDirFile, AppConfigFilename)
        appConfigFile.writeText(jsonString)
    }

    private fun addModelConfig(modelConfig: ModelConfig, modelUrl: String, isBuiltin: Boolean) {
        require(!localIdSet.contains(modelConfig.localId))
        localIdSet.add(modelConfig.localId)
        modelList.add(
            ModelState(
                modelConfig,
                modelUrl,
                File(appCacheFile, modelConfig.localId)
            )
        )
        if (!isBuiltin) {
            updateAppConfig {
                appConfig.modelList.add(ModelRecord(modelUrl, modelConfig.localId, modelConfig.modelLib))
            }
        }
    }

    private fun deleteModel(localId: String) {
        val modelDirFile = File(appCacheFile, localId)
        modelDirFile.deleteRecursively()
        require(!modelDirFile.exists())
        localIdSet.remove(localId)
        modelList.removeIf { modelState -> modelState.modelConfig.localId == localId }
        updateAppConfig {
            appConfig.modelList.removeIf { modelRecord -> modelRecord.localId == localId }
        }
    }

    private fun isModelConfigAllowed(modelConfig: ModelConfig): Boolean {
        if (appConfig.modelLibs.contains(modelConfig.modelLib)) return true
        viewModelScope.launch {
            issueAlert("Model lib ${modelConfig.modelLib} is not supported.")
        }
        return false
    }

    inner class ModelState(
        val modelConfig: ModelConfig,
        private val modelUrl: String,
        private val modelDirFile: File
    ) {
        var modelInitState = mutableStateOf(ModelInitState.Initializing)
        private var paramsConfig = ParamsConfig(emptyList())
        val progress = mutableStateOf(0.0)
        val total = mutableStateOf(1)
        val id: UUID = UUID.randomUUID()
        private val remainingTasks = emptySet<DownloadTask>().toMutableSet()
        private val downloadingTasks = emptySet<DownloadTask>().toMutableSet()
        private val maxDownloadTasks = 3
        private val gson = Gson()


        init {
            switchToInitializing()
        }

        private fun switchToInitializing() {
            val paramsConfigFile = File(modelDirFile, ParamsConfigFilename)

            loadParamsConfig()
            switchToIndexing()
        }

        private fun loadParamsConfig() {
            val paramsConfigFile = File(modelDirFile, ParamsConfigFilename)
            require(paramsConfigFile.exists())
            val jsonString = paramsConfigFile.readText()
            paramsConfig = gson.fromJson(jsonString, ParamsConfig::class.java)
        }

        fun handleStart() {
            switchToDownloading()
        }

        fun handlePause() {
            switchToPausing()
        }

        fun handleClear() {
            require(
                modelInitState.value == ModelInitState.Downloading ||
                        modelInitState.value == ModelInitState.Paused ||
                        modelInitState.value == ModelInitState.Finished
            )
            switchToClearing()
        }

        private fun switchToClearing() {
            if (modelInitState.value == ModelInitState.Paused) {
                modelInitState.value = ModelInitState.Clearing
                clear()
            } else if (modelInitState.value == ModelInitState.Finished) {
                modelInitState.value = ModelInitState.Clearing
                if (chatState.modelName.value == modelConfig.localId) {
                    chatState.requestTerminateChat { clear() }
                } else {
                    clear()
                }
            } else {
                modelInitState.value = ModelInitState.Clearing
            }
        }

        fun handleDelete() {
            require(
                modelInitState.value == ModelInitState.Downloading ||
                        modelInitState.value == ModelInitState.Paused ||
                        modelInitState.value == ModelInitState.Finished
            )
            switchToDeleting()
        }

        private fun switchToDeleting() {
            if (modelInitState.value == ModelInitState.Paused) {
                modelInitState.value = ModelInitState.Deleting
                delete()
            } else if (modelInitState.value == ModelInitState.Finished) {
                modelInitState.value = ModelInitState.Deleting
                if (chatState.modelName.value == modelConfig.localId) {
                    chatState.requestTerminateChat { delete() }
                } else {
                    delete()
                }
            } else {
                modelInitState.value = ModelInitState.Deleting
            }
        }

        private fun switchToIndexing() {
            modelInitState.value = ModelInitState.Indexing
            progress.value = 0.0
            total.value = paramsConfig.paramsRecords.size + 19 // param_shard_0.bin is too huge
//            total.value = modelConfig.tokenizerFiles.size + paramsConfig.paramsRecords.size
//            for (tokenizerFilename in modelConfig.tokenizerFiles) {
//                val file = File(modelDirFile, tokenizerFilename)
//                if (file.exists()) {
//                    ++progress.value
//                } else {
//                    remainingTasks.add(
//                        DownloadTask(
//                            URL("${modelUrl}${ModelUrlSuffix}${tokenizerFilename}"),
//                            file
//                        )
//                    )
//                }
//            }
            for (paramsRecord in paramsConfig.paramsRecords) {
                val file = File(modelDirFile, paramsRecord.dataPath)
                var value = 1
                if (paramsRecord.dataPath == "params_shard_0.bin")
                    value = 20
                if (file.exists()) {
                    progress.value += value
                } else {
                    remainingTasks.add(
                        DownloadTask(
                            "${modelUrl}$ModelUrlSuffix${paramsRecord.dataPath}",
                            file
                        )
                    )
                }
            }
            if (progress.value < total.value) {
                switchToPaused()
            } else {
                switchToFinished()
            }
        }

        var timer = Timer()
        val task_map = mutableMapOf<Long, DownloadTask>()
        var large_task_id: Long = -1
        var last_large_progress: Long = 0

        inner class Reciever: BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                val task_id = intent.getLongExtra(DownloadManager.EXTRA_DOWNLOAD_ID, -1)

                val name = task_map[task_id]!!.file.name
                File(modelDirFile.parentFile, "Download/${name}").renameTo(File("${modelDirFile}/${name}"))
                handleFinishDownload(task_map[task_id]!!)
            }
        }

        inner class ProgressTask : TimerTask() {
            @SuppressLint("Range")
            public override fun run() {
                val downloadQuery = DownloadManager.Query()
                downloadQuery.setFilterById(large_task_id)
                val cursor = downloadManager.query(downloadQuery);
                if (cursor != null && cursor.moveToFirst()) {
                    val total = cursor.getLong(cursor.getColumnIndex(DownloadManager.COLUMN_TOTAL_SIZE_BYTES))
                    val cur = cursor.getLong(cursor.getColumnIndex(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR))
                    progress.value += (cur - last_large_progress).toFloat() / total * 19
                    last_large_progress = cur
                    cursor.close()
                }
            }
        }

        private fun switchToDownloading() {
            val onDownloaded = Reciever()
            application.registerReceiver(onDownloaded, IntentFilter(DownloadManager.ACTION_DOWNLOAD_COMPLETE))
            timer.schedule(ProgressTask(), 0, 1000)

            modelInitState.value = ModelInitState.Downloading
            for (downloadTask in remainingTasks) {
                if (downloadingTasks.size < maxDownloadTasks) {
                    handleNewDownload(downloadTask)
                } else {
                    return
                }
            }
        }

        private fun handleNewDownload(downloadTask: DownloadTask) {
            require(modelInitState.value == ModelInitState.Downloading)
            require(!downloadingTasks.contains(downloadTask))
            downloadingTasks.add(downloadTask)

            val request = DownloadManager.Request(Uri.parse(downloadTask.url)).setDestinationInExternalFilesDir(application, Environment.DIRECTORY_DOWNLOADS, downloadTask.file.name)
            val task_id = downloadManager.enqueue(request)
            task_map[task_id] = downloadTask
            if (downloadTask.file.name == "params_shard_0.bin") {
                large_task_id = task_id
            }
        }

        private fun handleNextDownload() {
            require(modelInitState.value == ModelInitState.Downloading)
            for (downloadTask in remainingTasks) {
                if (!downloadingTasks.contains(downloadTask)) {
                    handleNewDownload(downloadTask)
                    break
                }
            }
        }

        private fun handleFinishDownload(downloadTask: DownloadTask) {
            remainingTasks.remove(downloadTask)
            downloadingTasks.remove(downloadTask)
            ++progress.value
            require(
                modelInitState.value == ModelInitState.Downloading ||
                        modelInitState.value == ModelInitState.Pausing ||
                        modelInitState.value == ModelInitState.Clearing ||
                        modelInitState.value == ModelInitState.Deleting
            )
            if (modelInitState.value == ModelInitState.Downloading) {
                if (remainingTasks.isEmpty()) {
                    if (downloadingTasks.isEmpty()) {
                        timer.cancel()
                        switchToFinished()
                    }
                } else {
                    handleNextDownload()
                }
            } else if (modelInitState.value == ModelInitState.Pausing) {
                if (downloadingTasks.isEmpty()) {
                    switchToPaused()
                }
            } else if (modelInitState.value == ModelInitState.Clearing) {
                if (downloadingTasks.isEmpty()) {
                    clear()
                }
            } else if (modelInitState.value == ModelInitState.Deleting) {
                if (downloadingTasks.isEmpty()) {
                    delete()
                }
            }
        }

        private fun clear() {
            val files = modelDirFile.listFiles { dir, name ->
                !(dir == modelDirFile && !name.startsWith("params"))
            }
            require(files != null)
            for (file in files) {
                file.deleteRecursively()
                require(!file.exists())
            }
            File(modelDirFile.parent!!+"/Download").deleteRecursively()
            val modelConfigFile = File(modelDirFile, ModelConfigFilename)
            require(modelConfigFile.exists())
            switchToIndexing()
        }

        private fun delete() {
            modelDirFile.deleteRecursively()
            require(!modelDirFile.exists())
            requestDeleteModel(modelConfig.localId)
        }

        private fun switchToPausing() {
            modelInitState.value = ModelInitState.Pausing
        }

        private fun switchToPaused() {
            modelInitState.value = ModelInitState.Paused
        }


        private fun switchToFinished() {
            modelInitState.value = ModelInitState.Finished
        }

        fun startChat() {
            chatState.requestReloadChat(
                modelConfig.localId,
                modelConfig.modelLib,
                modelDirFile.absolutePath
            )
        }

    }

    inner class ChatState {
        val messages = emptyList<MessageData>().toMutableStateList()
        val report = mutableStateOf("")
        val modelName = mutableStateOf("")
        private var modelChatState = mutableStateOf(ModelChatState.Ready)
            @Synchronized get
            @Synchronized set
        private val backend = ChatModule()
        private var modelLib = ""
        private var modelPath = ""
        private val executorService = Executors.newSingleThreadExecutor()
        private var has_user_prompt = false
        private var is_first_ask = true

        private fun mainResetChat() {
            is_first_ask = true
            has_user_prompt = false
            executorService.submit {
                callBackend { backend.resetChat() }
                viewModelScope.launch {
                    clearHistory()
                    switchToReady()
                }
            }
        }

        private fun clearHistory() {
            messages.clear()
            report.value = ""
        }


        private fun switchToResetting() {
            modelChatState.value = ModelChatState.Resetting
        }

        private fun switchToGenerating() {
            modelChatState.value = ModelChatState.Generating
        }

        private fun switchToReloading() {
            modelChatState.value = ModelChatState.Reloading
        }

        private fun switchToReady() {
            modelChatState.value = ModelChatState.Ready
        }

        private fun switchToFailed() {
            modelChatState.value = ModelChatState.Falied
        }

        private fun callBackend(callback: () -> Unit): Boolean {
            try {
                callback()
            } catch (e: Exception) {
                viewModelScope.launch {
                    val stackTrace = e.stackTraceToString()
                    val errorMessage = e.localizedMessage
                    appendMessage(
                        MessageRole.Bot,
                        "MLCChat failed\n\nStack trace:\n$stackTrace\n\nError message:\n$errorMessage"
                    )
                    switchToFailed()
                }
                return false
            }
            return true
        }

        fun requestResetChat() {
            require(interruptable())
            interruptChat(
                prologue = {
                    switchToResetting()
                },
                epilogue = {
                    mainResetChat()
                }
            )
        }

        private fun interruptChat(prologue: () -> Unit, epilogue: () -> Unit) {
            // prologue runs before interruption
            // epilogue runs after interruption
            require(interruptable())
            if (modelChatState.value == ModelChatState.Ready) {
                prologue()
                epilogue()
            } else if (modelChatState.value == ModelChatState.Generating) {
                prologue()
                executorService.submit {
                    viewModelScope.launch { epilogue() }
                }
            } else {
                require(false)
            }
        }

        fun requestTerminateChat(callback: () -> Unit) {
            require(interruptable())
            interruptChat(
                prologue = {
                    switchToTerminating()
                },
                epilogue = {
                    mainTerminateChat(callback)
                }
            )
        }

        private fun mainTerminateChat(callback: () -> Unit) {
            executorService.submit {
                callBackend { backend.unload() }
                viewModelScope.launch {
                    clearHistory()
                    switchToReady()
                    callback()
                }
            }
        }

        private fun switchToTerminating() {
            modelChatState.value = ModelChatState.Terminating
        }


        fun requestReloadChat(modelName: String, modelLib: String, modelPath: String) {
            if (this.modelName.value == modelName && this.modelLib == modelLib && this.modelPath == modelPath) {
                return
            }
            require(interruptable())
            interruptChat(
                prologue = {
                    switchToReloading()
                },
                epilogue = {
                    mainReloadChat(modelName, modelLib, modelPath)
                }
            )
        }

        private fun mainReloadChat(modelName: String, modelLib: String, modelPath: String) {
            clearHistory()
            this.modelName.value = modelName
            this.modelLib = modelLib
            this.modelPath = modelPath
            executorService.submit {
                viewModelScope.launch {
                    Toast.makeText(application, "Initialize...", Toast.LENGTH_LONG).show()
                }
                if (!callBackend {
                        backend.unload()
                        backend.reload(modelLib, modelPath)
                    }) return@submit
                viewModelScope.launch {
                    Toast.makeText(application, "Ready to chat", Toast.LENGTH_SHORT).show()
                    switchToReady()
                }
            }
        }

        fun requestImage(imgs: Array<IntArray>, height: Int, width: Int, cols: Int): Future<*>? {
            //require(chatable())
            var newText = ""
            switchToGenerating()
            Toast.makeText(application, "Image Processing...", Toast.LENGTH_SHORT).show()
            var f = executorService.submit {
                var steps = 0
                for (img in imgs) {
                    report.value = "Image processing $steps, $height, $width"
                    var new_line = 0
                    if (steps > 0 && steps % cols == 0){
                        new_line = 1
                    }
                    callBackend { backend.image(img, steps, height, width, new_line) }
                    steps += 1
                    if (steps >= 2) {break}
                }
                has_user_prompt = true
                viewModelScope.launch {
                    report.value = "Image $steps process is done, ask any question"
                    if (modelChatState.value == ModelChatState.Generating) switchToReady()
                }
            }
            return f
        }

        fun requestGenerate(prompt: String, steps: Int) {
            require(chatable())
            var newText = ""
            switchToGenerating()
            executorService.submit {
                appendMessage(MessageRole.User, prompt)
                appendMessage(MessageRole.Bot, "")
                if (has_user_prompt) {
                    if (steps == 0){
                        if (!callBackend { backend.prefill("</image>" + prompt + "<AI>") }) return@submit
                    }else{
                        if (!callBackend { backend.prefill("</image></slice>" + prompt + "<AI>") }) return@submit
                    }
                    has_user_prompt = false
                } else {
                    if (is_first_ask && !modelName.value.endsWith("-V")) {
                        if (!callBackend { backend.prefill("<系统命令>你是MiniCPM，由面壁智能开发。<用户>" + prompt + "<AI>") }) return@submit
                        is_first_ask = false;
                    }
                    else {
                        if (!callBackend { backend.prefill("<用户>" + prompt + "<AI>") }) return@submit
                    }
                }
                while (!backend.stopped()) {
                    if (!callBackend {
                            backend.decode()
                            newText = backend.message
                            viewModelScope.launch { updateMessage(MessageRole.Bot, newText) }
                        }) return@submit
                    if (modelChatState.value != ModelChatState.Generating) return@submit
                }
                val runtimeStats = backend.runtimeStatsText()
                viewModelScope.launch {
                    report.value = runtimeStats
                    if (modelChatState.value == ModelChatState.Generating) switchToReady()
                }
            }
        }

        private fun appendMessage(role: MessageRole, text: String) {
            messages.add(MessageData(role, text))
        }


        private fun updateMessage(role: MessageRole, text: String) {
            messages[messages.size - 1] = MessageData(role, text)
        }

        fun chatable(): Boolean {
            return modelChatState.value == ModelChatState.Ready
        }

        fun interruptable(): Boolean {
            return modelChatState.value == ModelChatState.Ready
                    || modelChatState.value == ModelChatState.Generating
                    || modelChatState.value == ModelChatState.Falied
        }
    }
}

enum class ModelInitState {
    Initializing,
    Indexing,
    Paused,
    Downloading,
    Pausing,
    Clearing,
    Deleting,
    Finished
}

enum class ModelChatState {
    Generating,
    Resetting,
    Reloading,
    Terminating,
    Ready,
    Falied
}

enum class MessageRole {
    Bot,
    User
}

data class DownloadTask(val url: String, val file: File)

data class MessageData(val role: MessageRole, val text: String, val id: UUID = UUID.randomUUID(), var image_path: String = "")

data class AppConfig(
    @SerializedName("model_libs") val modelLibs: List<String>,
    @SerializedName("model_list") val modelList: MutableList<ModelRecord>,
)

data class ModelRecord(
    @SerializedName("model_url") val modelUrl: String,
    @SerializedName("local_id") val localId: String,
    @SerializedName("model_lib") val modelLib: String
)

data class ModelConfig(
    @SerializedName("model_lib") var modelLib: String,
    @SerializedName("local_id") var localId: String,
    @SerializedName("tokenizer_files") val tokenizerFiles: List<String>
)

data class ParamsRecord(
    @SerializedName("dataPath") val dataPath: String
)

data class ParamsConfig(
    @SerializedName("records") val paramsRecords: List<ParamsRecord>
)