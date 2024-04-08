package com.modelbest.mlcllm;

import org.apache.tvm.Device;
import org.apache.tvm.Function;
import org.apache.tvm.Module;
import org.apache.tvm.NDArray;
import org.apache.tvm.TVMType;

public class ChatModule {
    private Function reloadFunc;
    private Function unloadFunc;
    private Function prefillFunc;
    private Function imageFunc;
    private Function decodeFunc;
    private Function getMessage;
    private Function stoppedFunc;
    private Function resetChatFunc;
    private Function runtimeStatsTextFunc;
    private Module llmChat;

    public ChatModule() {
        Function createFunc = Function.getFunction("mlc.llm_chat_create");
        assert createFunc != null;
        llmChat = createFunc.pushArg(Device.opencl().deviceType).pushArg(0).invoke().asModule();
        reloadFunc = llmChat.getFunction("reload");
        unloadFunc = llmChat.getFunction("unload");
        prefillFunc = llmChat.getFunction("prefill");
        imageFunc = llmChat.getFunction("image");
        decodeFunc = llmChat.getFunction("decode");
        getMessage = llmChat.getFunction("get_message");
        stoppedFunc = llmChat.getFunction("stopped");
        resetChatFunc = llmChat.getFunction("reset_chat");
        runtimeStatsTextFunc = llmChat.getFunction("runtime_stats_text");
    }

    public void unload() {
        unloadFunc.invoke();
    }

    public void reload(String modelLib, String modelPath) {
        String libPrefix = modelLib.replace('-', '_') + "_";
        Function systemLibFunc = Function.getFunction("runtime.SystemLib");
        assert systemLibFunc != null;
        systemLibFunc = systemLibFunc.pushArg(libPrefix);
        Module lib = systemLibFunc.invoke().asModule();
        reloadFunc = reloadFunc.pushArg(lib).pushArg(modelPath);
        reloadFunc.invoke();
    }

    public void resetChat() {
        resetChatFunc.invoke();
    }

    public void prefill(String input) {
        prefillFunc.pushArg(input).invoke();
    }

    public void image(int[] inp, int steps, int height, int width) {
        //int C = 3, H = 224, W = 224;
        int C = 3, H = height, W = width;
        long[] shape = {1, C, H, W};
        NDArray img = NDArray.empty(shape, new TVMType("int32"));
        img.copyFrom(inp);
        //imageFunc.pushArg(img).invoke();
        imageFunc.pushArg(img).pushArg(steps).pushArg(height).pushArg(width).invoke();
    }

    public String getMessage() {
        return getMessage.invoke().asString();
    }

    public String runtimeStatsText() {
        return runtimeStatsTextFunc.invoke().asString();
    }

    public void evaluate() {
        llmChat.getFunction("evaluate").invoke();
    }

    public boolean stopped() {
        return stoppedFunc.invoke().asLong() != 0L;
    }

    public void decode() {
        decodeFunc.invoke();
    }
}